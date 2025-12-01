# SPR²Q: Static Priority-based Rectifier Routing Quantization for Image Super-Resolution

This project is developed based on the **PaddlePaddle** framework.

## Environment Setup
```bash
cd SPR2Q
conda create -n SPR2Q python=3.9
conda activate SPR2Q
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install -r requirements.txt
```

## Dataset Preparation

This repository uses DF2K for training and Set5, Set14, B100, Urban100, and Manga109 for testing. Please modify the dataset paths in the YAML configuration files located in options/train/, options/test/, and options/finetune/.

## Training Instructions

For example, to train 2× super-resolution with 4-bit quantization, modify the pretrain_network_FP in the YAML file to point to the full-precision model (.pdparams), then run:
```bash
python basicsr/train.py -opt options/train/train_mamba_quant_x2.yml --force_yml bit=4 name=train_x2_bit4
```
The training results will be saved in the experiments folder.

## Fine-Tuning for Weight Factor Adjustment

Modify pretrain_network_FP to the full-precision model (.pdparams) and pretrain_network_Q to the trained quantized model (.pdparams) you want to fine-tune. Then run:
```bash
python basicsr/train.py -opt options/finetune/finetune_mamba_quant_x2.yml --force_yml bit=1 name=finetune_x2_bit4
```
The fine-tuned results will be saved in the experiments folder.

## Testing Instructions

Modify pretrain_network_Q in the YAML file to the trained quantized model (.pdparams). Then run:
```bash
python basicsr/test.py -opt options/test/test_mamba_quant_x2.yml --force_yml bit=4 name=test_x2_bit4
```
The test results will be saved in the results folder.

