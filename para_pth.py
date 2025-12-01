import paddle
import torch
import numpy as np
import numpy

def convert_paddle_to_pytorch(paddle_path, pytorch_path):
    
    # 1. 加载 Paddle 检查点
    try:
        checkpoint = paddle.load(paddle_path)
        print(f"成功加载 Paddle 检查点: {paddle_path}") 
        
        # 提取 state_dict
        if 'params' in checkpoint:
            paddle_state_dict = checkpoint['params']
        elif 'model' in checkpoint:
            paddle_state_dict = checkpoint['model']
        else: 
            paddle_state_dict = checkpoint 
            
    except Exception as e:
        print(f"错误: 无法加载 Paddle 权重文件: {e}")
        return

    # 2. 创建新的 PyTorch state_dict
    pytorch_state_dict = {}
    print("开始转换...")

    # 3. 遍历 Paddle 权重
    for paddle_key, paddle_tensor in paddle_state_dict.items():
        
        # 确保是张量
        if not isinstance(paddle_tensor, paddle.Tensor):
            print(f"  [警告] 键 '{paddle_key}' 不是张量，跳过。")
            continue
            
        # --- 步骤 3.1: 键名映射 (BN 层) ---
        if '_mean' in paddle_key:
            pytorch_key = paddle_key.replace('_mean', 'running_mean')
        elif '_variance' in paddle_key:
            pytorch_key = paddle_key.replace('_variance', 'running_var')
        else:
            pytorch_key = paddle_key

        
        # --- 步骤 3.2: 权重张量转换 (Tensor Conversion) ---
        
        numpy_data = paddle_tensor.numpy()
        original_shape = numpy_data.shape

        # 规则 1：'lora' 在键名中的不要转置 (来自你的修正)
        if 'lora' in paddle_key:
            print(f"  [跳过转置] 侦测到 LoRA 键: {paddle_key}")
            pass # 保持原样

        elif "selectiveScan.x_proj_weight" in paddle_key:
            if len(original_shape) == 2:
                numpy_data = numpy_data[np.newaxis, :, :]  # [48, 19] -> [1, 19, 48]
            elif len(original_shape) == 3:
                numpy_data = numpy_data.transpose(0, 2, 1)
            print(f"  [转换] selectiveScan.x_proj_weight 转置: {paddle_key}")

        elif "selectiveScan.dt_projs_weight" in paddle_key:
            if len(original_shape) == 2:
                numpy_data = numpy_data[np.newaxis, :, :]  # [3, 48] -> [1, 48, 3]
            elif len(original_shape) == 3:
                numpy_data = numpy_data.transpose(0, 2, 1)
            print(f"  [转换] selectiveScan.dt_projs_weight 转置: {paddle_key}")

        elif "selectiveScan.dt_projs_bias" in paddle_key and len(original_shape) == 1:
            numpy_data = np.expand_dims(numpy_data, axis=0)  # [48] -> [1, 48]
            print(f"  [转换] selectiveScan.dt_projs_bias Unsqueeze: {paddle_key}")

        elif "relative_position_bias_table" in paddle_key:
            numpy_data = numpy_data
            print(f"  [转换] relative_position_bias_table 转置: {paddle_key}")

        elif "embeddingA.weight" in paddle_key or "embeddingB.weight" in paddle_key:
            numpy_data = numpy_data
            print(f"  [转换] embedding 权重转置: {paddle_key}")

        elif "A_logs" in paddle_key:
            numpy_data = numpy_data
            print(f"  [转换] A_logs 转置: {paddle_key}")

        # 默认 2D 层（Linear 等）
        elif len(original_shape) == 2:
            numpy_data = numpy_data.T
            print(f"  [转换] 默认2D转置: {paddle_key}")
            
        # 规则 6：修复标量 (scalars)
        if paddle_key.endswith(('.lower_bound', '.upper_bound', '.n_bit')):
            if original_shape == ():  # 检查是否为标量
                print(f"  [转换] Reshaping 标量 (scalar) -> (1,): {paddle_key}")
                numpy_data = numpy_data.reshape(1) 
        
        # ======================================================

        # C. 转换为 PyTorch 张量
        # 使用 .copy() 确保解除内存占用
        pytorch_tensor = torch.from_numpy(numpy_data.copy())
        
        pytorch_state_dict[pytorch_key] = pytorch_tensor
        print(f"  转换: {paddle_key} (shape={paddle_tensor.shape}) -> {pytorch_key} (shape={pytorch_tensor.shape})")

    # 4. 保存为 PyTorch 检查点文件
    try:
        # 重新添加 'params' 包装
        final_checkpoint_dict = {'params': pytorch_state_dict}
        torch.save(final_checkpoint_dict, pytorch_path)
        print(f"\n转换成功! PyTorch 权重已保存到: {pytorch_path} (已添加 'params' 包装)")
    except Exception as e:
        print(f"\n错误: 无法保存 PyTorch 权重文件: {e}")

# --- 如何使用 ---
PADDLE_MODEL_PATH = '/data1/LWH/paddlepaddle_train/experiments/train_paddle_x2_bit2/models/net_Q_11900.pdparams' # 你的 Paddle 权重文件
PYTORCH_MODEL_PATH = '/data1/LWH/paddlepaddle_train/experiments/train_paddle_x2_bit2/models/net_Q_11900.pth' # 你想保存的 PyTorch 权重文件

convert_paddle_to_pytorch(PADDLE_MODEL_PATH, PYTORCH_MODEL_PATH)

# --- (推荐) 验证步骤 ---
# from your_pytorch_model_def import YourPyTorchModel # 导入你的 PyTorch 模型类

# print("\n正在验证加载 PyTorch 模型...")
# try:
#     torch_model = YourPyTorchModel()
#     torch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))
#     torch_model.eval()
#     print("✅ 成功加载权重到 PyTorch 模型。")
# except Exception as e:
#     print(f"❌ 验证失败: PyTorch 模型加载 state_dict 时出错: {e}")
#     print("这通常意味着你的'键名映射'不正确，或者权重维度转换错误。")