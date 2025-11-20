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

# pylint: disable=invalid-name, unused-argument
"""FFI for tvm.node
# 该模块是TVM中"node"模块（主要对应IR节点、 runtime对象等带属性的核心对象）的FFI（外部函数接口）
# 核心作用：
# 1. 为"运行时仅模式（runtime only mode）"提供基础函数的默认实现（降级逻辑）
# 2. 通过TVM的FFI机制，将C++层以"node."为前缀注册的全局函数，自动绑定到当前Python模块
# 3. 实现Python与C++底层"node相关功能"（如对象字符串表示、属性访问、序列化）的交互
"""
import tvm._ffi


# The implementations below are default ones when the corresponding
# functions are not available in the runtime only mode.
# They will be overriden via _init_api to the ones registered
# via TVM_REGISTER_GLOBAL in the compiler mode.
def AsRepr(obj):
    '''运行时模式下，生成对象的字符串表示（默认实现）'''
    return obj.type_key() + "(" + obj.handle.value + ")"


def AsLegacyRepr(obj):
    return obj.type_key() + "(" + obj.handle.value + ")"


def NodeListAttrNames(obj):
    return lambda x: 0


def NodeGetAttr(obj, name):
    raise AttributeError()


def SaveJSON(obj):
    raise RuntimeError("Do not support object serialization in runtime only mode")


def LoadJSON(json_str):
    raise RuntimeError("Do not support object serialization in runtime only mode")


# Exports functions registered via TVM_REGISTER_GLOBAL with the "node" prefix.
# e.g. TVM_REGISTER_GLOBAL("node.AsRepr")-> 绑定为 Python 的 tvm.node.AsRepr。
tvm._ffi._init_api("node", __name__)
