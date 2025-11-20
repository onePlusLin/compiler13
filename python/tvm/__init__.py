# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=redefined-builtin, wildcard-import
# 顶层初始化文件，整合核心组件，定义顶层接口，并设置异常处理机制，为用户提供统一的入口
"""TVM: Open Deep Learning Compiler Stack."""
import multiprocessing
import sys
import os
import traceback


# ------------------------------
# 1. 导入TVM底层核心组件（C扩展接口层）
# ------------------------------
# top-level alias
# tvm._ffi
# TVMError: TVM框架统一异常基类；__version__: TVM版本号；_RUNTIME_ONLY: 是否仅启用运行时（无编译功能）
from ._ffi.base import TVMError, __version__, _RUNTIME_ONLY

# 导入数据类型相关定义：DataTypeCode标识数据类型类别（如INT、FLOAT），DataType描述具体数据类型（如int32、float64）
# 导入注册机制相关函数：用于注册自定义对象、函数和扩展
from ._ffi.runtime_ctypes import DataTypeCode, DataType
from ._ffi import register_object, register_func, register_extension, get_global_func



# ------------------------------
# 2. 导入运行时（Runtime）核心组件
# ------------------------------
# top-level alias
# tvm.runtime
from .runtime.object import Object
from .runtime.ndarray import device, cpu, cuda, gpu, opencl, cl, vulkan, metal, mtl
from .runtime.ndarray import vpi, rocm, ext_dev, hexagon
from .runtime import ndarray as nd # 为runtime.ndarray模块设置别名nd，方便用户通过tvm.nd访问（如tvm.nd.array创建张量）



# ------------------------------
# 3. 导入核心功能模块（IR、TIR、编译驱动等）
# ------------------------------
# 错误处理模块：包含TVM自定义异常类型（如DiagnosticError）
# tvm.error
from . import error

# tvm.ir
# IR（Intermediate Representation，中间表示）相关模块
from .ir import IRModule        # IR模块容器（存放Relay/TIR计算图）
from .ir import transform       # IR转换/优化工具（如算子融合、内存优化pass）
from .ir import instrument      # IR instrumentation（用于调试、性能分析的插桩工具）
from .ir import container       # IR容器类型（如Array、Map，用于管理IR节点集合）
# 内存池相关配置：定义内存池属性、工作区/常量内存池管理
from .ir import PoolInfo
from .ir import WorkspacePoolInfo
from .ir import ConstantPoolInfo
from .ir import PoolInfoProperties
from .ir import WorkspaceMemoryPools
from .ir import ConstantMemoryPools
from . import ir                 # 导出整个ir模块，支持用户通过tvm.ir访问所有子组件


# tvm.tir
from . import tir

# tvm.target
from . import target

# tvm.te
from . import te

# tvm.driver
# Driver模块：编译驱动核心接口（build编译模型、lower将计算图 lowering 到TIR）？？？？
from .driver import build, lower

# tvm.parser
# Parser模块：用于解析TVM文本格式的IR（如将Relay代码字符串解析为IR对象）
from . import parser

# others
# ------------------------------
# 4. 导入辅助与扩展模块
# ------------------------------
# Arith模块：算术分析与简化工具（如常量折叠、表达式化简）
from . import arith

# support infra
# Support模块：框架辅助工具（如读取编译配置、环境检查）
from . import support

# Contrib initializers 
# 第三方硬件支持初始化（仅导入，不直接暴露给用户）
# rocm: AMD GPU支持；nvcc: NVIDIA CUDA编译工具；sdaccel: Xilinx FPGA支持,vela!
from .contrib import rocm as _rocm, nvcc as _nvcc, sdaccel as _sdaccel


# 条件导入Micro模块（微型设备支持，如嵌入式芯片）
# 仅当：1. 非纯运行时模式（_RUNTIME_ONLY=False）；2. 编译时启用了MICRO支持（USE_MICRO=ON）
if not _RUNTIME_ONLY and support.libinfo().get("USE_MICRO", "OFF") == "ON":
    from . import micro

# NOTE: This file should be python2 compatible so we can
# raise proper error message when user run the package using
# an older version of the python

# ------------------------------
# 5. 配置全局异常处理（增强用户体验与系统稳定性）
# ------------------------------
def _should_print_backtrace():
    """判断是否需要打印异常回溯信息（用于控制异常输出详细程度）"""
    # 1. 若在pytest测试环境（环境变量PYTEST_CURRENT_TEST存在），强制打印回溯
    in_pytest = "PYTEST_CURRENT_TEST" in os.environ
    # 2. 读取环境变量TVM_BACKTRACE（0=不打印，1=打印），默认不打印
    tvm_backtrace = os.environ.get("TVM_BACKTRACE", "0")

    try:
        # 将TVM_BACKTRACE值转换为布尔值（仅支持0/1）
        tvm_backtrace = bool(int(tvm_backtrace))
    except ValueError:
        # 若环境变量值非法，抛出明确错误
        raise ValueError(
            f"invalid value for TVM_BACKTRACE {tvm_backtrace}, please set to 0 or 1."
        )

    # 满足任一条件即打印回溯：测试环境 或 显式开启TVM_BACKTRACE
    return in_pytest or tvm_backtrace


def tvm_wrap_excepthook(exception_hook):
    """包装系统默认的异常钩子（excepthook），添加TVM专属异常处理逻辑
    参数：
        exception_hook: 系统默认的异常处理函数（sys.excepthook）
    返回：
        wrapper: 增强后的异常处理函数
    """
    def wrapper(exctype, value, trbk):
        """异常处理包装函数，实现两个核心功能：
        1. 对TVM诊断性错误（DiagnosticError）提供友好提示；
        2. 异常发生时终止所有残留子进程，避免资源泄漏。
        """
        # 处理TVM诊断性错误：未开启回溯时，提示用户如何开启详细信息
        if exctype is error.DiagnosticError and not _should_print_backtrace():
            print("note: run with `TVM_BACKTRACE=1` environment variable to display a backtrace.")
        else:
            # 其他异常：调用系统默认钩子打印完整回溯
            exception_hook(exctype, value, trbk)

        # 异常发生后，终止所有活跃子进程（防止多进程编译时残留进程占用资源）
        if hasattr(multiprocessing, "active_children"):
            for p in multiprocessing.active_children():
                p.terminate()

    return wrapper


# 替换系统默认的异常钩子为TVM增强版，全局生效
sys.excepthook = tvm_wrap_excepthook(sys.excepthook)
