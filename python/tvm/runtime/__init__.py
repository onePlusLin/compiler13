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
"""TVM runtime namespace.
该模块定义了TVM运行时的核心组件，包含模型执行、数据处理、设备交互等必备功能，
是TVM编译后模型加载、运行和结果处理的入口。
"""

# class exposures
# ------------------------------
# 导出核心类（运行时基础组件）
# ------------------------------
from .packed_func import PackedFunc                             # 跨语言函数封装类，支持Python与C++函数互调
from .object import Object                                      # TVM所有对象的基类（C++层Object的Python包装，通过句柄交互）
from .object_path import ObjectPath, ObjectPathPair             # 对象路径类，用于追踪对象在IR中的位置（调试/分析用）
from .script_printer import Scriptable                          # 支持脚本打印的接口类，使对象可转换为TVM脚本字符串
from .object_generic import ObjectGeneric, ObjectTypes          # 通用对象处理类，提供Python类型与TVM对象的适配
from .ndarray import NDArray, DataType, DataTypeCode, Device    # 张量与设备相关类：
# NDArray：TVM张量数据结构（包装C++ DLTensor，通过句柄管理）
# DataType：数据类型描述（如float32、int64）
# DataTypeCode：数据类型编码（标识类型类别，如INT、FLOAT）
# Device：设备信息（如CPU、GPU，包含设备类型和设备ID）
from .module import Module, num_threads                         # 模块与线程配置类：
# Module：TVM模块（可加载编译后的模型，封装C++ Module对象）
# num_threads：设置运行时线程数
from .profiling import Report                                   # 性能分析报告类，用于存储和展示模型运行时的性能数据（如耗时、内存占用）

# function exposures
# ------------------------------
# 导出核心函数（运行时功能接口）
# ------------------------------
from .object_generic import (
    convert_to_object,          # 将Python对象转换为TVM Object
    convert,                    # 通用转换函数，自动将Python类型转为对应的TVM类型
    const                       # 创建TVM常量对象
)
from .ndarray import (
    device,                     # 根据设备类型和ID获取设备上下文
    cpu,                        # 获取CPU设备上下文（快捷函数）
    cuda, gpu,                  # 获取CUDA/GPU设备上下文（快捷函数）
    opencl, cl,                 # 获取OpenCL设备上下文（快捷函数）
    vulkan,                     # 获取Vulkan设备上下文（快捷函数）
    metal, mtl,                 # 获取Metal设备上下文（快捷函数）
    vpi,                        # 获取VPI设备上下文（快捷函数，用于FPGA等）
    rocm,                       # 获取ROCM设备上下文（快捷函数，用于AMD GPU）
    ext_dev                     # 获取扩展设备上下文（快捷函数，用于自定义设备）
)
from .module import (
    load_module,                # 加载编译后的TVM模块（如.so、.tar等格式）
    enabled,                    # 检查指定设备类型是否可用（如检查CUDA是否启用）
    system_lib,                 # 获取TVM系统库模块（包含预编译的通用函数）
    load_static_library         # 加载静态链接库作为TVM模块
)
from .container import (
    String,                     # TVM字符串容器（包装C++字符串，支持跨语言传递）
    ShapeTuple                  # 形状元组容器（存储张量维度信息，支持TVM IR中的形状操作）
)
from .params import (
    save_param_dict,            # 将模型参数字典序列化为字节流
    load_param_dict,            # 从字节流反序列化为模型参数字典
    save_param_dict_to_file,    # 将参数字典保存到文件
    load_param_dict_from_file   # 从文件加载参数字典
)

from . import executor          # 执行器模块，包含模型运行时的执行逻辑（如GraphExecutor、VMExecutor等）
