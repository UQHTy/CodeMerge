# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time
import os
import torch.nn.functional as F
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmcv.runner import save_checkpoint
from torch import nn
import torch.optim as optim
from mmcv.parallel import scatter
import os
from pathlib import Path
from mmdet.core import encode_mask_results
from sklearn.cluster import KMeans
from mmdet.models import build_detector
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
from mmcv.runner import build_optimizer
from fvcore.nn import FlopCountAnalysis, flop_count_table

SAVE_CKPT = [0,2, 4,8, 16,2100]
SAVE_CKPT_INTERVAL = 16


def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order="F", dtype="uint8"
                )
            )[0]
        )  # encoded with RLE
    return [encoded_mask_results]

def loadCheckpoint_intoModel(checkpoint, model):
    """_summary_

    Args:
        checkpoint (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    modelParameters_names = set(checkpoint.keys())
    modelStateDict_keys = set(model.state_dict().keys())
    # breakpoint()
    assert modelParameters_names.issubset(modelStateDict_keys)
    # Must tie the encoder and decoder embeddings to the shared weight if the shared weight is a parameter.
    if "transformer.shared.weight" in checkpoint:
        checkpoint["transformer.decoder.embed_tokens.weight"] = checkpoint[
            "transformer.shared.weight"
        ]
        checkpoint["transformer.encoder.embed_tokens.weight"] = checkpoint[
            "transformer.shared.weight"
        ]

    update_stats = model.load_state_dict(checkpoint, strict=False)
    logger.info(f"Missing keys: {update_stats.missing_keys}")
    logger.info(f"Unexpected keys: {update_stats.unexpected_keys}")

    if "transformer.shared.weight" in checkpoint:
        del checkpoint["transformer.decoder.embed_tokens.weight"]
        del checkpoint["transformer.encoder.embed_tokens.weight"]

    return model

def aggregate_model(main_model,model_path_list, pretrained_model,indices,cfg_model,test_cfg):

    model_named_buffers = main_model.module.named_buffers() if hasattr(main_model,'module') else main_model.named_buffers()
    agg_model = None
    weight_i = 0
    agg_model = []
    ptm_check= pretrained_model.state_dict()
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys=[])
    sorted_numbers = {int(num) for key in model_path_list.keys() for num in re.findall(r'\d+', key)}
    # breakpoint()
    if len(model_path_list)>5:
        # breakpoint()
        selected_keys = [list(model_path_list.keys())[i] for i in indices]
        numbers = [int(key.split('_')[0]) for key in selected_keys]
        sorted_numbers = sorted(numbers)
        
    print("sorted_numbers==",sorted_numbers)
    for model_path in sorted(sorted_numbers):
        
        # past_model = build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=dataset)
        model_path1 = 'XXX/ckpts/checkpoint_iter_' + str(model_path) + '.pth'
        # past_model.load_params_from_file(filename=model_path1, logger=logger,report_logger=False, to_cpu=dist)
        past_model = build_detector(cfg_model, test_cfg=test_cfg)
        checkpoint = load_checkpoint(past_model, model_path1, map_location="cpu")
        wrap_fp16_model(past_model)
        past_model.cuda()
        past_model.eval()
        pretrained_state_dict0=past_model.state_dict()
        agg_model.append(pretrained_state_dict0)
        del past_model; torch.cuda.empty_cache()
        

    # breakpoint()
    agg_model = loadCheckpoint_intoModel(new_state_dict, pretrained_model)


    for agg_bf, bf in zip(agg_model.named_buffers(), model_named_buffers):
        agg_name, agg_value = agg_bf
        name, value = bf
        assert agg_name == 'module.' + name, 'name not equal:{} , {}'.format(agg_name,
                                                                name)
        parts = name.split('.')
        if 'running_mean' in parts or 'running_var' in parts:
            agg_value.data = value.data
    return agg_model


class TestTimeModel(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.backbone = detector.extract_feat
        self.head     = detector.head

    def forward(self, imgs, metas):
        # 只关心检测分支的前向
        feats = self.backbone(imgs)
        det_out, _, _, _ = self.head(feats, metas)
        return det_out

def custom_multi_gpu_test(
    model,
    data_loader,
    tmpdir=None,
    gpu_collect=False,
    pseudo_det=False,
    pretrained_model = None,
    cfg_model = None,
    test_cfg = None

):
    # breakpoint()
    # … 前面不变 …
    pretrained_model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False

    if pseudo_det:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        det_params = list(model.module.head.det_head.parameters())
        optimizer = optim.AdamW(det_params, lr=5e-3, weight_decay=6e-9) 


    model_bank_feat = {}
    for i, data in enumerate(data_loader):

        cur_it = prog_bar.completed
        samples_seen = int(cur_it)*int(8)
        # ———— 伪标签微调分支 ————
        if pseudo_det:
            # a) scatter DataContainer -> tensors + metas
            device = torch.cuda.current_device()
            scattered = scatter(data, [device])[0]
            imgs = scattered['img']
            metas = scattered

            ckpt_dir = Path("XXX/Sparse_stage2/work_dirs/ckpts")
            pth_files = list(ckpt_dir.glob("*.pth"))
            count = len(pth_files)
            super_model = None
            if count >=3:
                shared_feat_keys = sorted(
                    [k for k in model_bank_feat.keys() if k.endswith("instance_features")],
                    key=lambda x: int(x.split('_')[0])
                )
                device = torch.device("cuda:0")
                W_rand = torch.randn(230400, 1024).to(device)
                RP = []
                for i, key_i in enumerate(shared_feat_keys):
                    shared_feature = model_bank_feat[key_i].reshape(1, 900*256).squeeze(0)
                    Features_h= shared_feature @ W_rand
                    RP.append(shared_feature.cpu().detach().numpy())


                super_model = aggregate_model(model,model_bank_feat, pretrained_model,indices,cfg_model,test_cfg)

            feats = super_model.module.extract_feat(imgs)#weak
            det_out, _, _, _ = super_model.module.head(feats, metas)

            feats0 = model.module.extract_feat(imgs) # main model strong aug imgs 
            det_out0, _, _, _ = model.module.head(feats0, metas)


            feats1 = pretrained_model.module.extract_feat(imgs)
            det_out1, _, _, _ = pretrained_model.module.head(feats1, metas)
            # breakpoint()
            # c) 从最后一层 decoder 取出 scores & boxes

            cls_scores = det_out['cls_score'][-1][0].detach()  # [Nq, num_cls]
            box_preds  = det_out['reg_pred']    [-1][0].detach()  # [Nq, D]
            # d) 各类伪标签选取：  
            probs, pred_labels = cls_scores.sigmoid().max(dim=1)  
                
                # f) 构造伪标签数据
            pseudo_data = [probs,pred_labels]

            # g) 截断 det_out 里所有 decoder 的输出到 D
            det_out_trunc = [det_out0['pred_labels'],det_out0['reg_pred'],det_out0['quality']]
  
            # h) 计算 loss 并反向更新
            optimizer.zero_grad()
            loss = model.head.loss(det_out_trunc, pseudo_data)
            print("loss==",loss)
            loss.backward()
            optimizer.step()

        #########################################################
        # breakpoint()
        if (samples_seen in SAVE_CKPT or samples_seen % SAVE_CKPT_INTERVAL==0):
            # breakpoint()
            ckpt_name = os.path.join(
                "XXXXX/Sparse_stage2/work_dirs/ckpts",
                f"checkpoint_iter_{samples_seen}.pth"
            )
            save_checkpoint(model, ckpt_name)
            model_bank_feat[str(samples_seen)+'_instance_features'] = det_out1["instance_feature"]


        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, dict):
            if 'bbox_results' in result:
                bbox_results.extend(result['bbox_results'])
                batch_size = len(result['bbox_results'])
            else:
                batch_size = len(result)
            if 'mask_results' in result and result['mask_results'] is not None:
                mr = custom_encode_mask_results(result['mask_results'])
                mask_results.extend(mr)
                have_mask = True
        else:
            batch_size = len(result)
            bbox_results.extend(result)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    avg_fps = total_images / (total_time_ms / 1000.0)
    print(f"Avg FPS (CUDA timing): {avg_fps:.2f}")
    # ———— 多卡结果汇总 ————
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        mask_results = (
            collect_results_gpu(mask_results, len(dataset))
            if have_mask else None
        )
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        mask_results = (
            collect_results_cpu(mask_results, len(dataset), tmpdir + '_mask')
            if have_mask else None
        )

    if mask_results is None:
        return bbox_results
    
    return {'bbox_results': bbox_results, 'mask_results': mask_results}
     

def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full(
            (MAX_LEN,), 32, dtype=torch.uint8, device="cuda"
        )
        if rank == 0:
            mmcv.mkdir_or_exist(".dist_test")
            tmpdir = tempfile.mkdtemp(dir=".dist_test")
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device="cuda"
            )
            dir_tensor[: len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f"part_{rank}.pkl"))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f"part_{i}.pkl")
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        """
        bacause we change the sample of the evaluation stage to make sure that
        each gpu will handle continuous sample,
        """
        # for res in zip(*part_list):
        for res in part_list:
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)
