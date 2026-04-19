"""C++ 高性能扑克引擎适配层

优先加载 C++ 编译的 .so 模块；若未编译则 fallback 到纯 Python 引擎。
编译方式: pip install pybind11 && python setup_engine.py build_ext --inplace
"""

USE_CC_ENGINE = False

try:
    from .poker_engine_cc import GameEngine, decode_action as cc_decode_action
    USE_CC_ENGINE = True
except ImportError:
    pass
