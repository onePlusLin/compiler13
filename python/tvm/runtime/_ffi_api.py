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
"""FFI APIs for tvm.runtime"""
import tvm._ffi

# 调用TVM FFI的API初始化函数，完成C++与Python的接口绑定
# 参数1: "runtime" —— C++层全局函数的命名空间前缀（仅匹配带此前缀的函数）
# 参数2: __name__ —— 当前Python模块的名称（将C++函数绑定到当前模块）
# 效果：C++中用TVM_REGISTER_GLOBAL("runtime.XXX")注册的函数，会自动成为当前模块的XXX函数
# Exports functions registered via TVM_REGISTER_GLOBAL with the "runtime" prefix.

# e.g. TVM_REGISTER_GLOBAL("runtime.ModuleLoadFromFile")→ 绑定为 Python 的 tvm.runtime.ModuleLoadFromFile。
tvm._ffi._init_api("runtime", __name__)
