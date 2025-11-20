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
"""C interfacing code.

This namespace contains everything that interacts with C code.
Most TVM C related object are ctypes compatible, which means
they contains a handle field that is ctypes.c_void_p and can
be used via ctypes function calls.

Some performance critical functions are implemented by cython
and have a ctypes fallback implementation.
# 1. TVM的C相关对象通常与ctypes兼容,即包含c_void_p类型的handle字段,可通过ctypes函数调用直接使用
# 2. 部分性能关键函数由cython实现,同时提供ctypes的降级兼容实现 fallback
"""
from . import _pyversion            # 导入Python版本兼容性相关模块（处理不同Python版本的差异，确保C接口在各版本中兼容）
from .base import register_error    # 从base模块导入错误注册函数（用于向TVM框架注册C层定义的错误类型，实现Python与C错误处理的对接）


# 从registry模块导入核心注册机制函数：
# register_object：注册C定义的对象类型，使Python能识别并操作C对象
# register_func：注册C定义的函数，使Python能调用C层函数
# register_extension：注册扩展类型，支持自定义C扩展与Python的交互
from .registry import register_object, register_func, register_extension

# 从registry模块导入API初始化与查询函数：
# _init_api：初始化C层导出的API，建立Python函数与C函数的映射
# get_global_func：获取TVM全局注册的C函数（通过函数名查找）
# get_object_type_index：获取C对象类型的索引（用于类型识别和转换）
from .registry import _init_api, get_global_func, get_object_type_index
