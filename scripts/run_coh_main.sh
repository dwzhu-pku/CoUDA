# debug_mode="-m debugpy --listen 127.0.0.1:6679"

CUDA_VISIBLE_DEVICES=2 python ${debug_mode} ./src/coh_main.py \
    --mode albert_both \
    --arch xxlarge \
    --aggregate avg  \
    --dataset summeval \
    --path_to_ckp ./models/couda_model