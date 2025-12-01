import logging
import paddle
import os
from os import path as osp
import sys
sys.path.append('.')
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline_fp32(root_path):
    # 1️⃣ 解析配置
    opt, _ = parse_options(root_path, is_train=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt['gpu'])

    # 2️⃣ 强制禁用量化相关逻辑
    if 'quant' in opt and opt['quant']:
        print("[Info] 已禁用量化模式进行全精度测试")
        opt['quant'] = False
    if 'model_type' in opt and 'Quant' in opt['model_type']:
        opt['model_type'] = opt['model_type'].replace('Quant', '')
        print(f"[Info] 修改模型类型为全精度: {opt['model_type']}")

    # 3️⃣ 设为确定性模式 + 全精度
    paddle.set_flags({'FLAGS_cudnn_deterministic': True})
    paddle.set_default_dtype('float32')

    # 4️⃣ 初始化日志与路径
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_fp32_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # 5️⃣ 创建测试数据集
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # 6️⃣ 构建模型（全精度）
    model = build_model(opt)
    print("[Info] 模型构建完成，全精度推理中...")

    # 7️⃣ 进行测试
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'🔍 Testing {test_set_name} (FP32)...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline_fp32(root_path)
