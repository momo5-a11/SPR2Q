import torch
import paddle
import numpy as np
import os
from collections import OrderedDict

def convert_pytorch_to_paddle(pth_path, pdparams_path):
    print(f"Loading PyTorch checkpoint: {pth_path}")
    ckpt = torch.load(pth_path, map_location='cpu')

    # --- 自动提取实际 state_dict ---
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            torch_state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            torch_state_dict = ckpt['model']
        else:
            torch_state_dict = ckpt
    elif isinstance(ckpt, OrderedDict):
        torch_state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    paddle_state_dict = {}

    for name, torch_tensor in torch_state_dict.items():
        # 清理前缀
        clean_name = name.replace("params.", "").replace("module.", "")

        if isinstance(torch_tensor, OrderedDict):
            for subname, subtensor in torch_tensor.items():
                sub_clean_name = f"{clean_name}.{subname}".replace("params.", "").replace("module.", "")
                tensor = subtensor.detach().cpu().numpy()

                # [修改后的转置逻辑]
                needs_transpose = 'weight' in sub_clean_name and tensor.ndim == 2 and \
                                  "embeddingA.weight" not in sub_clean_name and \
                                  "embeddingB.weight" not in sub_clean_name
                                  
                if needs_transpose:
                    print(f"  Transposing: {sub_clean_name}")
                    tensor = tensor.T
                else:
                    print(f"  Converting (no transpose): {sub_clean_name}")
                paddle_state_dict[sub_clean_name] = paddle.to_tensor(tensor)
        else:
            tensor = torch_tensor.detach().cpu().numpy()
            
            # [修改后的转置逻辑]
            needs_transpose = 'weight' in clean_name and tensor.ndim == 2 and \
                              "embeddingA.weight" not in clean_name and \
                              "embeddingB.weight" not in clean_name

            if needs_transpose:
                print(f"  Transposing: {clean_name}")
                tensor = tensor.T
            else:
                print(f"  Converting (no transpose): {clean_name}")
                
            paddle_state_dict[clean_name] = paddle.to_tensor(tensor)

    os.makedirs(os.path.dirname(pdparams_path), exist_ok=True)
    paddle.save(paddle_state_dict, pdparams_path)
    print(f"✅ Successfully converted to: {pdparams_path}")


if __name__ == "__main__":
    # ======== 修改这里 ========
    pytorch_pth = "/data1/LWH/MambaIR-main/experiments/pretrained_models/mambairv2_lightSR_x4.pth"         # 你的 PyTorch 模型路径
    paddle_pdparams = "/data1/LWH/MambaIR-main/experiments/pretrained_models/mambairv2_lightSR_x4.pdparams"  # 目标 Paddle 模型路径
    # ===========================

    convert_pytorch_to_paddle(pytorch_pth, paddle_pdparams)
