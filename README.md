```bash
cd SPR2Q
conda create -n SPR2Q python=3.9
conda activate SPR2Q
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install -r requirements.txt
```

## 数据集
包括DF2K（训练）和Set5、set14、B100、Urban100、Manga109（测试），在options/train/、options/test/、options/finetune/中配置文件中修改数据集位置

## 训练运行指令（以2倍放大4bit为例）
修改yml文件中的pretrain_network_FP为全精度模型的pdparams文件
python basicsr/train.py -opt options/train/train_mamba_quant_x2.yml --force_yml bit=4 name=train_x2_bit4  结果保存在experiments

## 加权因子调校运行指令
修改yml文件中的pretrain_network_FP为全精度模型的pdparams文件、pretrain_network_Q为训练好进行加权因子调校的pdparams文件
python basicsr/train.py -opt options/finetune/finetune_mamba_quant_x2.yml --force_yml bit=1 name=finetune_x2_bit4  结果保存在experiments

## 测试运行指令
修改yml文件中的pretrain_network_Q为训练好的pdparams文件
python basicsr/test.py -opt options/test/test_mamba_quant_x2.yml --force_yml bit=4 name=test_x2_bit4  结果保存在results