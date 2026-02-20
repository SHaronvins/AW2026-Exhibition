# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from setuptools import setup
import os,sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import load

code_dir = os.path.dirname(os.path.realpath(__file__))


def get_eigen_include_dirs():
    """Return candidate Eigen include dirs (header-only)."""
    candidates = []

    # User override.
    for key in ("EIGEN3_INCLUDE_DIR", "EIGEN_INCLUDE_DIR", "EIGEN_DIR"):
        value = os.getenv(key)
        if value:
            candidates.append(value)

    # Conda environment (Windows + Linux).
    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend([
            os.path.join(conda_prefix, "Library", "include", "eigen3"),
            os.path.join(conda_prefix, "Library", "include"),
            os.path.join(conda_prefix, "include", "eigen3"),
            os.path.join(conda_prefix, "include"),
        ])

    # Common system locations (kept for Linux compatibility).
    candidates.extend([
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ])

    # De-dup + keep only existing.
    seen = set()
    out = []
    for path in candidates:
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isdir(norm):
            out.append(norm)
    return out


if sys.platform == "win32":
    # MSVC flags (these are passed to cl.exe).
    c_flags = ["/O2", "/std:c++17"]
    # NVCC flags. Host compiler flags must be routed via -Xcompiler.
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-Xcompiler",
        "/O2",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
else:
    nvcc_flags = [
        "-Xcompiler",
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    c_flags = ["-O3", "-std=c++17"]

setup(
    name='common',
    ext_modules=[
        CUDAExtension('common', [
            'bindings.cpp',
            'common.cu',
        ],extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags}),
        CUDAExtension('gridencoder', [
            f"{code_dir}/torch_ngp_grid_encoder/gridencoder.cu",
            f"{code_dir}/torch_ngp_grid_encoder/bindings.cpp",
        ],extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags}),
    ],
    include_dirs=get_eigen_include_dirs(),
    cmdclass={
        'build_ext': BuildExtension
})
