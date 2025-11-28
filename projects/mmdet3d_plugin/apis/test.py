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
from .merge_utils import *
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
from mmcv.runner import build_optimizer
# SAVE_CKPT = [16,32, 2100]
# # SAVE_CKPT_INTERVAL = 80
# SAVE_CKPT_INTERVAL= 112

SAVE_CKPT = [0,2, 4,8, 16,32,616,656,672, 2100]
SAVE_CKPT_INTERVAL = 80


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
    # The encoder and decoder embedding tokens are always tied to the shared weight and so should never be a paremeter
    # assert set(
    #     [
    #         "transformer.decoder.embed_tokens.weight",
    #         "transformer.encoder.embed_tokens.weight",
    #     ]
    # ).issubset(modelStateDict_keys.difference(modelParameters_names))

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
    # breakpoint()
    # pretrained_model_path = '/home/huitong/YHDiskB/code/MOS/MOS_TTA_3D_DET/ckpts/waymo_pretrain/checkpoint_epoch_20.pth'
    # pretrained_model = build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=dataset)
    # pretrained_model.load_params_from_file(filename=pretrained_model_path, logger=logger,report_logger=False, to_cpu=dist)
    # pretrained_model.cuda()
    # pretrained_model.eval()

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
        model_path1 = '/home/uqhyan14/code/SparseDrive/work_dirs/ckpts/checkpoint_iter_' + str(model_path) + '.pth'
        # past_model.load_params_from_file(filename=model_path1, logger=logger,report_logger=False, to_cpu=dist)
        past_model = build_detector(cfg_model, test_cfg=test_cfg)
        checkpoint = load_checkpoint(past_model, model_path1, map_location="cpu")
        wrap_fp16_model(past_model)
        past_model.cuda()
        past_model.eval()
        # for name, param in past_model.named_parameters():
        #     param.data.mul_(model_weights[0].data)
        pretrained_state_dict0=past_model.state_dict()
        agg_model.append(pretrained_state_dict0)

    # ft_checks = [torch.load(fp) for fp in model_path_list]
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys=[]) for check in agg_model]
    )
    tv_flat_checks = flat_ft - flat_ptm
    reset_type = "topk"
    reset = "topk90"
    reset_thresh = eval(reset[len(reset_type) :])
    resolve = "mass"
    merge = "dis-mean"
    merged_tv = merge_methods(reset_type,tv_flat_checks,reset_thresh=reset_thresh,resolve_method=resolve,merge_func=merge,)
    # breakpoint()
    merged_check = flat_ptm +  0.85* merged_tv
    reference_state_dict = agg_model[0]
    merged_checkpoint = vector_to_state_dict(
                merged_check, reference_state_dict, remove_keys=[])
    new_state_dict = OrderedDict()
    for k, v in merged_checkpoint.items():
        new_key = 'module.' + k
        new_state_dict[new_key] = v
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
    model.eval()
    pretrained_model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    # if rank == 0:
    prog_bar = mmcv.ProgressBar(len(dataset))
        
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False

    if pseudo_det:
        model.train()
        # for m in model.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()

        # —— 3) 解冻 backbone 中的所有 BatchNorm2d 的 weight 和 bias，
        #     并保证它们在 train 模式下更新 running_mean/var
        # import torch.nn as nn
        # for m in model.module.img_backbone.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.train()  # 让 BN 更新 running stats
        #         if m.weight is not None:
        #             m.weight.requires_grad = True
        #         if m.bias is not None:
        #             m.bias.requires_grad = True

        # —— 4) 同样，你还可以解冻 neck 或 det_head 的全部参数
        # （这里我只举 backbone BN 的例子，det_head/ne​ck 根据需要自己控制）
        det_params      = list(model.module.head.det_head.parameters())
        neck_params     = list(model.module.img_neck.parameters())
        img_backbone_params = list(model.module.img_backbone.parameters())
        trainable_params = det_params + neck_params + img_backbone_params
        # —— 5) 最终把所有 requires_grad=True 的参数拼到一起
        # trainable_params = []
        # for p in det_params + neck_params + list(model.module.img_backbone.parameters()):
        #     if p.requires_grad:
        #         trainable_params.append(p)
        optimizer = optim.AdamW(trainable_params, lr=1e-7, weight_decay=1e-3) #5e-6,1e-6
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

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
            # breakpoint()

            ckpt_dir = Path("/home/uqhyan14/code/SparseDrive/work_dirs/ckpts")
            pth_files = list(ckpt_dir.glob("*.pth"))
            count = len(pth_files)
            super_model = None
            if count >=3:
                shared_feat_keys = sorted(
                    [k for k in model_bank_feat.keys() if k.endswith("instance_features")],
                    key=lambda x: int(x.split('_')[0])
                )
                # device = torch.device("cuda:0")
                
                device = model_bank_feat['0_instance_features'].device
                # breakpoint()
                W_rand = torch.randn(230400, 1024).to(device)
                RP = []
                for i, key_i in enumerate(shared_feat_keys):
                    shared_feature = model_bank_feat[key_i].reshape(1, 900*256).squeeze(0).to(device)
                    Features_h= shared_feature @ W_rand
                    # RP.append(shared_feature.cpu().detach().numpy())
                    RP.append(Features_h)

                indices = None
                if len(RP)>5:
                    K = 5
                    lambda_damping = 1e-3 
                    # breakpoint()
                    fingerprints = torch.stack(RP, dim=0) # [N,1024]
                    F = fingerprints - fingerprints.mean(dim=0, keepdim=True) #[N, 1024]
                    N, D = F.shape
                    H = (F.T @ F) / N #[1024, 1024])
                    H += lambda_damping * torch.eye(D, dtype=F.dtype, device=F.device) #[1024, 1024])
                    H_inv = torch.linalg.inv(H) #[1024, 1024])
                    scores = (F @ H_inv * F).sum(dim=1) #[N]
                    values,indices = torch.topk(scores, k=5)
                    del values

                super_model = aggregate_model(model,model_bank_feat, pretrained_model,indices,cfg_model,test_cfg)

            # b) forward 拿到 det_out
            if super_model is not None:
                print("SUPER_MODEL tta")
                # breakpoint()
                feats = super_model.module.extract_feat(imgs)#weak
                det_out, _, _, _ = super_model.module.head(feats, metas)
            else:
                feats = model.module.extract_feat(imgs)
                det_out, _, _, _ = model.module.head(feats, metas)

            model.eval()
            feats0 = model.module.extract_feat(imgs) # main model strong aug imgs 
            det_out0, _, _, _ = model.module.head(feats0, metas)


            feats1 = pretrained_model.module.extract_feat(imgs)
            det_out1, _, _, _ = pretrained_model.module.head(feats1, metas)
            # breakpoint()
            # c) 从最后一层 decoder 取出 scores & boxes
            pseudo_boxes3d = []
            pseudo_labels3d = []
            for i in range(6):
                cls_scores = det_out['classification'][i][0].detach()  # [Nq, num_cls]
                box_preds  = det_out['prediction']    [i][0].detach()  # [Nq, D]
                # d) 各类伪标签选取：  
                probs, pred_labels = cls_scores.sigmoid().max(dim=1)  

                keep = probs > 0.6             # 选所有类别，只要置信度 > 0.6  
                pseudo_boxes  = box_preds[keep]       # [M, D]  
                pseudo_labels = pred_labels[keep]      # [M]  
                
                sampler = model.module.head.det_head.sampler
                D = box_preds.shape[-1]

                sampler.cls_wise_reg_weights = {}  # 如果不想做 class-specific weighting

                pseudo_boxes = pseudo_boxes[..., :D-1]
                pseudo_boxes3d.append(pseudo_boxes)
                pseudo_labels3d.append(pseudo_labels)
                # f) 构造伪标签数据
            pseudo_data = {
                'gt_bboxes_3d': pseudo_boxes3d,
                'gt_labels_3d': pseudo_labels3d,
            }

            # g) 截断 det_out 里所有 decoder 的输出到 D
            det_out_trunc = {
                'classification': det_out0['classification'],
                'prediction':    det_out0['prediction'],
                'quality':       det_out0['quality'],
            }
            model.train()
            # breakpoint()

            # h) 计算 loss 并反向更新
            optimizer.zero_grad()
            loss_dict = model.module.head.det_head.loss(det_out_trunc, pseudo_data)
            loss = sum(loss_dict.values())
            # print("loss_dict==",loss_dict)
            print("loss==",loss)
            loss.backward()

            optimizer.step()
        # ———— 常规推理 & 收集 ————
        model.eval()
        # breakpoint()
        if (samples_seen in SAVE_CKPT or samples_seen % SAVE_CKPT_INTERVAL==0):
            # breakpoint()
            # ckpt_name = "/home/uqhyan14/clip_retrain/end-to-end/test/SparseDrive/work_dirs/ckpts" / ('checkpoint_iter_%d' % samples_seen)
            ckpt_name = os.path.join(
                "/home/uqhyan14/code/SparseDrive/work_dirs/ckpts",
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

# def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
#     """Test model with multiple gpus.
#     This method tests model with multiple gpus and collects the results
#     under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
#     it encodes results to gpu tensors and use gpu communication for results
#     collection. On cpu mode it saves the results on different gpus to 'tmpdir'
#     and collects them by the rank 0 worker.
#     Args:
#         model (nn.Module): Model to be tested.
#         data_loader (nn.Dataloader): Pytorch data loader.
#         tmpdir (str): Path of directory to save the temporary results from
#             different gpus under cpu mode.
#         gpu_collect (bool): Option to use either gpu or cpu to collect results.
#     Returns:
#         list: The prediction results.
#     """
#     model.eval()
#     bbox_results = []
#     mask_results = []
#     dataset = data_loader.dataset
#     rank, world_size = get_dist_info()
#     if rank == 0:
#         prog_bar = mmcv.ProgressBar(len(dataset))
#     time.sleep(2)  # This line can prevent deadlock problem in some cases.
#     have_mask = False
#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=True, **data)
#             # encode mask results
#             # breakpoint()
#             if isinstance(result, dict):
#                 if "bbox_results" in result.keys():
#                     bbox_result = result["bbox_results"]
#                     batch_size = len(result["bbox_results"])
#                     bbox_results.extend(bbox_result)
#                 if (
#                     "mask_results" in result.keys()
#                     and result["mask_results"] is not None
#                 ):
#                     mask_result = custom_encode_mask_results(
#                         result["mask_results"]
#                     )
#                     mask_results.extend(mask_result)
#                     have_mask = True
#             else:
#                 batch_size = len(result)
#                 bbox_results.extend(result)

#         if rank == 0:
#             for _ in range(batch_size * world_size):
#                 prog_bar.update()

#     # collect results from all ranks
#     if gpu_collect:
#         bbox_results = collect_results_gpu(bbox_results, len(dataset))
#         if have_mask:
#             mask_results = collect_results_gpu(mask_results, len(dataset))
#         else:
#             mask_results = None
#     else:
#         bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
#         tmpdir = tmpdir + "_mask" if tmpdir is not None else None
#         if have_mask:
#             mask_results = collect_results_cpu(
#                 mask_results, len(dataset), tmpdir
#             )
#         else:
#             mask_results = None

#     if mask_results is None:
#         return bbox_results
#     return {"bbox_results": bbox_results, "mask_results": mask_results}

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
