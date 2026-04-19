"""编译 C++ 扑克引擎为 Python .so 模块

用法:
    pip install pybind11
    python setup_engine.py build_ext --inplace

编译完成后会在 poker_ai/engine/ 目录下生成 poker_engine_cc*.so，
之后 `from poker_ai.engine import USE_CC_ENGINE` 即可自动检测。
"""

import os
import sys
from setuptools import setup, Extension

# ---- pybind11 头文件路径 ----
try:
    import pybind11
    pybind11_includes = [pybind11.get_include(), pybind11.get_include(user=True)]
except ImportError:
    print("ERROR: pybind11 未安装，请先执行: pip install pybind11")
    sys.exit(1)

# ---- 编译标志 ----
extra_compile_args = ['-std=c++17', '-O3', '-march=native', '-ffast-math', '-Wall']
if sys.platform == 'darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.14']

# ---- 扩展模块 ----
ext = Extension(
    name='poker_ai.engine.poker_engine_cc',
    sources=[os.path.join('poker_ai', 'engine', 'poker_engine.cpp')],
    include_dirs=pybind11_includes,
    extra_compile_args=extra_compile_args,
    language='c++',
)

setup(
    name='poker_engine_cc',
    version='1.0.0',
    description='High-performance C++ poker engine for SEED RL training',
    ext_modules=[ext],
    zip_safe=False,
)
