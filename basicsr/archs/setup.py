# setup.py (最终确认版)
import os
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_csrc_paths():
    csrc_root = "./csrc"
    cpp_paths = [os.path.join(root, f) for root, _, files in os.walk(csrc_root) for f in files if f.endswith('.cpp')]
    cu_paths = [os.path.join(root, f) for root, _, files in os.walk(csrc_root) for f in files if f.endswith('.cu')]
    return cpp_paths, cu_paths

all_cpp_paths, all_cu_paths = get_csrc_paths()

setup(
    name='mambairv2light_arch',
    version='1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            'quant_cuda',
            sources=all_cpp_paths + all_cu_paths,
            include_dirs=[
                './csrc',
            ],
            extra_compile_args={
                'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
                'nvcc': [
                    '-O3', 
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    # ===============================================================
                    # 关键修改：请务必确认这里是 c++14，而不是 c++17
                    '-std=c++14',
                    # ===============================================================
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)