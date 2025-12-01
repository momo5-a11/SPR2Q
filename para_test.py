import paddle

pdparams_path = "/data1/LWH/MambaIR-main/experiments/pretrained_models/mambairv2_lightSR_x4.pdparams"  # 改成你的文件路径

state_dict = paddle.load(pdparams_path)

print(f"Loaded Paddle parameters from: {pdparams_path}")
print(f"Total parameters: {len(state_dict)}\n")

for i, (k, v) in enumerate(state_dict.items()):
    print(f"{i:04d}: {k:<80}  shape={list(v.shape)}")
    if i > 50:  # 只打印前 50 个，防止太长
        print("... (more not shown)")
        break

pdparams_path = "/data1/LWH/MambaIR-main/experiments/pretrained_models/mambairv2_lightSR_x2_final.pdparams"  # 改成你的文件路径

state_dict = paddle.load(pdparams_path)

print(f"Loaded Paddle parameters from: {pdparams_path}")
print(f"Total parameters: {len(state_dict)}\n")

for i, (k, v) in enumerate(state_dict.items()):
    print(f"{i:04d}: {k:<80}  shape={list(v.shape)}")
    if i > 50:  # 只打印前 50 个，防止太长
        print("... (more not shown)")
        break