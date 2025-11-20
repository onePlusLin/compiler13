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
# pylint: disable=invalid-name, unused-import
"""Runtime Object API
# 该模块定义TVM运行时对象的核心基类`Object`，是所有TVM运行时对象（如NDArray、Module、IR节点等）的父类
# 核心作用：通过FFI机制连接Python层对象与C++底层对象，提供统一的属性访问、序列化、哈希等接口
"""
import ctypes

from tvm._ffi.base import _FFI_MODE, _LIB, _RUNTIME_ONLY, c_str, check_call
from tvm._ffi.runtime_ctypes import ObjectRValueRef

# 导入TVM运行时FFI API：
# _ffi_api: 通用运行时FFI接口
# _ffi_node_api: 节点（Node）相关FFI接口（用于IR节点属性访问等）
from . import _ffi_api, _ffi_node_api

try:
    # pylint: disable=wrong-import-position,unused-import
    if _FFI_MODE == "ctypes":
        raise ImportError()
    from tvm._ffi._cy3.core import (
        ObjectBase,
        PyNativeObject,
        _set_class_object,
        _set_class_object_generic,
    )
except (RuntimeError, ImportError) as error:
    # pylint: disable=wrong-import-position,unused-import
    if _FFI_MODE == "cython":
        raise error
    from tvm._ffi._ctypes.object import ObjectBase, PyNativeObject
    from tvm._ffi._ctypes.packed_func import _set_class_object, _set_class_object_generic

# cls 是 Python 中 “类本身” 的引用
def _new_object(cls):
    """Helper function for pickle
        pickle序列化/反序列化的辅助函数，用于创建空对象实例
    """
    return cls.__new__(cls)


class Object(ObjectBase):
    """Base class for all tvm's runtime objects."""
    """所有TVM运行时对象的基类，封装C++层的Object对象，提供Python层统一接口
    
    核心特性：
    1. 通过`handle`属性（C++对象句柄）与底层交互
    2. 支持属性访问（自动映射到C++对象的属性）
    3. 支持序列化（pickle）、哈希、相等性判断
    """
    # 限制动态添加属性，避免与C++层属性冲突
    __slots__ = []

    def __repr__(self):
        """返回对象的字符串表示（用于调试，调用C++层API生成）"""
        return _ffi_node_api.AsRepr(self)

    def legacy_repr(self):
        """返回兼容旧版本的对象字符串表示"""
        return _ffi_node_api.AsLegacyRepr(self)

    def __dir__(self):
        """返回对象的所有属性名（结合Python类属性和C++层节点属性）
        
        用于支持Python的dir()函数，同时暴露C++层定义的对象属性
        """
        class_names = dir(self.__class__)
        fnames = _ffi_node_api.NodeListAttrNames(self)
        size = fnames(-1)
        return sorted([fnames(i) for i in range(size)] + class_names)

    def __getattr__(self, name):
        # specially check handle since
        # this is required for PackedFunc calls
        if name == "handle":
            raise AttributeError("handle is not set")

        try:
            return _ffi_node_api.NodeGetAttr(self, name)
        except AttributeError:
            raise AttributeError(f"{type(self)} has no attribute {name}") from None

    def __hash__(self):
        """返回对象的哈希值（基于C++层对象指针，确保唯一性）"""
        return _ffi_api.ObjectPtrHash(self)

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __reduce__(self):
        cls = type(self)
        return (_new_object, (cls,), self.__getstate__())

    def __getstate__(self):
        handle = self.handle
        if handle is not None:
            return {"handle": _ffi_node_api.SaveJSON(self)}
        return {"handle": None}

    def __setstate__(self, state):
        # pylint: disable=assigning-non-slot, assignment-from-no-return
        handle = state["handle"]
        self.handle = None
        if handle is not None:
            self.__init_handle_by_constructor__(_ffi_node_api.LoadJSON, handle)

    def _move(self):
        """Create an RValue reference to the object and mark the object as moved.

        This is a advanced developer API that can be useful when passing an
        unique reference to an Object that you no longer needed to a function.

        A unique reference can trigger copy on write optimization that avoids
        copy when we transform an object.

        Note
        ----
        All the reference of the object becomes invalid after it is moved.
        Be very careful when using this feature.

        Examples
        --------

        .. code-block:: python

           x = tvm.tir.Var("x", "int32")
           x0 = x
           some_packed_func(x._move())
           # both x0 and x will points to None after the function call.

        Returns
        -------
        rvalue : The rvalue reference.
        """
        return ObjectRValueRef(self)


_set_class_object(Object)
