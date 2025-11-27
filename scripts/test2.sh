bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    work_dirs/ckpts/checkpoint_iter_960.pth \
    1 \
    --deterministic \
    --pseudo-det \
    --eval bbox
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl
    # ckpt/sparsedrive_stage2.pth \