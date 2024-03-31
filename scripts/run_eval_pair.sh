for dataset in insted_cnn insted_wiki; do
    for model in albert; do
        CUDA_VISIBLE_DEVICES=2 python src/eval_pair.py --model ${model} --dataset ${dataset}
    done
done