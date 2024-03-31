# CoUDA

This repository contains the code, data and model for the paper "CoUDA: Coherence Evaluation via Unified Data Augmentation" (NAACL 2024)

## 🔧 Reproduction
To replicate our results, follow these steps to download the code and necessary dependencies:
```
git clone https://github.com/dwzhu-pku/CoUDA.git
cd CoUDA
pip install -r requirements.txt
```

The datasets used for training and evaluation can be found under `data/train` and `data/eval`, respectively.

For our trained model, please download it via this [link](https://drive.google.com/file/d/1BD8tV_rYu_mx0U4ASaeFiCIsdo6RXgJc/view?usp=sharing), and place it under `models/couda_model`. Note that the results reported in the paper is averaged across 3 runs with different random seeds. The uploaded one is selected in random, which turns out to have slightly lower score.

To calculate correlation scores with human ratings, you may run:

```
bash scripts/run_coh_main.sh
```

To calcuate pair-wise ranking scores, you may run:

```
bash scripts/run_eval_pair.sh
```


## 🌟 Citation
If you find this repo helpful, please cite our paper as follows:

```bibtex

```