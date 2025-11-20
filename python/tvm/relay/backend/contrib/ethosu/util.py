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
# pylint: disable=invalid-name
"""
Helper utility Enums and Functions used through out code generation.

The rest of the utility functions are misc.
Refer to the description inside such functions
"""

from inspect import signature
from enum import Enum
from typing import Union, Tuple, List
import numpy as np  # type: ignore

import tvm  # type: ignore
from tvm import relay
from tvm._ffi import register_object
from tvm.runtime import Object
from . import _ffi_api


class QConv2DArgs(Enum):
    """
    This is a helper enum to obtain the correct index
    of qnn.conv2d arguments.
    """

    IFM = 0
    WEIGHTS = 1
    IFM_ZERO_POINT = 2
    WEIGHTS_ZERO_POINT = 3
    IFM_SCALE = 4
    WEIGHTS_SCALE = 5


class QConv2DTransposeArgs(Enum):
    """
    This is a helper enum to obtain the correct index
    of qnn.conv2d_transpose arguments.
    """

    IFM = 0
    WEIGHTS = 1
    IFM_ZERO_POINT = 2
    WEIGHTS_ZERO_POINT = 3
    IFM_SCALE = 4
    WEIGHTS_SCALE = 5


class RequantArgs(Enum):
    """
    This is a helper enum to obtain the correct index
    of qnn.requantize arguments.
    """

    IFM_SCALE = 1
    IFM_ZERO_POINT = 2
    OFM_SCALE = 3
    OFM_ZERO_POINT = 4


class BiasAddArgs(Enum):
    """
    This is a helper enums to obtain the correct index
    of qnn.bias_add arguments.
    """

    BIASES = 1


class ClipArgs(Enum):
    """
    This is a helper enums to obtain the correct index
    of clip arguments.
    """

    A_MIN = 1
    A_MAX = 2


class BinaryElementwiseArgs(Enum):
    """This is a helper enums to access the correct index
    of binary elementwise arguments
    """

    IFM = 0
    IFM2 = 1
    IFM_SCALE = 2
    IFM_ZERO_POINT = 3
    IFM2_SCALE = 4
    IFM2_ZERO_POINT = 5
    OFM_SCALE = 6
    OFM_ZERO_POINT = 7


class QuantizeArgs(Enum):
    """
    This is a helper enums to access the correct index of
    quantize arguments
    """

    IFM = 0
    OFM_SCALE = 1
    OFM_ZERO_POINT = 2


class DequantizeArgs(Enum):
    """
    This is a helper enums to access the correct index of
    dequantize arguments
    """

    IFM = 0
    IFM_SCALE = 1
    IFM_ZERO_POINT = 2


class QDenseArgs(Enum):
    """
    This is a helper enum to access the correct index of
    qnn.dense arguments
    """

    IFM = 0
    WEIGHTS = 1
    IFM_ZERO_POINT = 2
    WEIGHTS_ZERO_POINT = 3
    IFM_SCALE = 4
    WEIGHTS_SCALE = 5


class QPadArgs(Enum):
    """
这是一个辅助枚举函数来获取nn.pad 变量的正确索引
    This is a helper enum to obtain the correct index
    of nn.pad arguments.
    """

    IFM = 0
    IFM_ZERO_POINT = 1


def is_npu_func(func: relay.Function) -> bool:#也是只检查@后面那个函数
    print("查看function的全部属性:\n",dir(func))
    print("is_nou_func:现在检查的这个\n：",func)
    """Check if the given function is an NPU function."""
    print("is_nou_func_func.attrs:\n",func.attrs)
    return func.attrs and "Compiler" in func.attrs and func.attrs["Compiler"] == "ethos-u"


def is_composite_func(func: relay.Function, name: str) -> bool:
    """
    该方法检查调用是否为给定名称的复合函数。
    This method checks whether the call is to
    a composite function of a given name.

    Parameters
    ----------
    func : relay.Function
        The header to be displayed along with the dump.

    name : str
        The candidate name to be checked

    Returns
    --------
    a boolean
    """

    if not hasattr(func, "attrs"):
        return False
    if "Composite" not in func.attrs.keys():
        return False
    composite_name = func.attrs["Composite"]

    return composite_name == name


def is_named_ethosu_op(expr: tvm.relay.Expr, name: str) -> bool:
    """Checks whether a relay expression matches that of the
    named operator.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The expression to check.
    name : str
        The name of the expected operator
        (without NPU prefix "contrib.ethosu").

    Returns
    -------
    bool
        True if expression matches name, false if not.
    """
    prefix = "contrib.ethosu."
    return (
        isinstance(expr, tvm.relay.expr.Call)
        and isinstance(expr.op, tvm.ir.op.Op)
        and expr.op.name == prefix + name
    )


def get_range_for_dtype_str(dtype: str) -> Tuple[int, int]:
    """
    Produce the min,max for a give data type.

    Parameters
    ----------
    dtype : str
        a type string (e.g., int8)

    Returns
    -------
    type_info.min : int
        the minimum of the range
    type_info.max : int
        the maximum of the range
    """

    try:
        type_info = np.iinfo(dtype)
    except ValueError:
        type_info = np.finfo(dtype)
    return type_info.min, type_info.max


def round_away_zero(f: Union[float, np.double, np.single, np.float32, np.float64]) -> np.float64:
    """Round the number away from zero towards +inf / -inf"""
    offset = -0.5 if (f < 0) else 0.5
    return np.trunc(f + offset)


def round_up(a: int, b: int) -> int:
    """Round up to a multiple of b"""
    return ((a + b - 1) // b) * b


def get_accelerator_config():
    """Get the variant of the accelerator to compile for"""
    compiler_attrs = tvm.get_global_func("relay.ext.ethos-u.get_compiler_attrs")()
    return compiler_attrs.accelerator_config


def is_cascader_enabled() -> bool:
    """Determine whether the cascader is enabled"""
    compiler_attrs = tvm.get_global_func("relay.ext.ethos-u.get_compiler_attrs")()
    return bool(compiler_attrs.enable_cascader)


def is_copying_constants_disabled() -> bool:
    """Determine whether copying constants is disabled for case without cascader"""
    compiler_attrs = tvm.get_global_func("relay.ext.ethos-u.get_compiler_attrs")()
    return bool(compiler_attrs.disable_copying_constants)


def is_striping_enabled() -> bool:
    """Determine whether the cascader is enabled"""
    compiler_attrs = tvm.get_global_func("relay.ext.ethos-u.get_compiler_attrs")()
    return bool(compiler_attrs.enable_striping)


def get_arg_count(func):
    """Helper function to get the number of
    arguments in a python function"""
    sig = signature(func)
    return len(sig.parameters)


def get_dim_value(layout: str, dim: int):
    """This is a helper function to retrieve the value
    of the dimension given the shape and the layout
    """
    assert isinstance(layout, str)
    assert dim in list(layout)
    for idx, dim_char in enumerate(layout):
        if dim_char == dim:
            return idx
    return None


def calculate_size_bytes(expr):
    """This is a helper function to calculate the number
    of bytes required to hold the tensor/relay.expr"""
    try:
        type_info = np.iinfo(expr.checked_type.dtype)
    except ValueError:
        type_info = np.finfo(expr.checked_type.dtype)
    element_size = type_info.bits // 8
    elements = np.prod(list(expr.checked_type.shape))
    return element_size * elements


@register_object("relay.ext.ethos-u.BaseAddress")
class BaseAddress(Object):
    """
    保存驱动程序所需指针基地址的结构
    This is a structure to hold base addresses for pointers
    provided for the driver.
    """

    def __init__(
        self,
        name: str,
        primfunc_param_idx: int,
        region: int,
        size: int,
        is_runtime_allocation: bool = False,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.BaseAddress,  # type: ignore # pylint: disable=no-member
            name,
            primfunc_param_idx,
            region,
            size,
            is_runtime_allocation,
        )


@register_object("relay.ext.ethos-u.CompilationArtifact")
class CompilationArtifact(Object):
    """
    保存microNPU二进制工件的结构
    This is a structure to hold binary artifacts
    for the microNPU.
    """

    def __init__(
        self,
        function_name: str,
        command_stream: str,
        encoded_constants: str,
        base_addresses: List[BaseAddress],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.CompilationArtifact,  # type: ignore # pylint: disable=no-member
            function_name,
            command_stream,
            encoded_constants,
            base_addresses,
        )

# zejia
def create_npu_function_pass(opt_level: int, name: str = ""):
    """
    将一个pass类装饰为一个NPU函数pass，但是要在里面定义transform_npu_function，
    这个trans...函数是真正用来作用mod里面的NPU函数的！
    `transform_npu_function(global_variable, relay_function)`
    A utility decorator that wraps a given class as an NPU function pass. That is,
    a pass that behaves like a function pass and only traverses NPU external
    functions. How each NPU function is mutated is defined by the
    `transform_npu_function(global_variable, relay_function)` function which should
    be created in the class that is to be decorated. See the example below.

    Example
    -------
    This small example demonstrates a pass over NPU functions that performs no
    mutation.

    @create_npu_function_pass(opt_level=1)
    class MyPass:
        def transform_npu_function(self, global_var, func):
            return func

    mod = tvm.IRModule()
    mod = MyPass()(mod)

    Parameters
    ----------
    opt_level: int
        Optimization level for the module pass.
    name: str, optional
        Name for the module pass.

    Returns
    -------
    decorator
        The npu_pass decorator.
    """

    def decorator(npu_pass_class):
        @tvm.ir.transform.module_pass(name=name, opt_level=opt_level)
        class ModulePassWrapper:
            """The wrapper for the NPU pass."""

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def transform_module(self, mod: tvm.ir.IRModule, _) -> tvm.ir.IRModule:
                # --------------------------------------
                '''
                mod是整一个IRModule
                这个函数会筛选出@tvm_default...的NPU函数，对其进行回调npu_pass_class这个函数的pass处理
                '''
                # print("------------create_npu_function_pass strat---------------")
                # print("create_npu_function_pass:mod:",mod)
                # print("create_npu_function_pass:mod里面都有啥：\n")
                # print("查看mod的全部属性：\n",dir(mod))
                
                # print(f"模块中有 {len(mod.functions)} 个函数:")
                # for global_var, func in mod.functions.items():
                #     print("--------------------")
                #     # 1. 打印函数名 (GlobalVar)
                #     print(f"函数名 (GlobalVar): {global_var.name_hint}")

                #     # 2. 打印函数的属性 (Attributes)
                #     # 这非常重要！你可以看到这个函数是不是被分配给了 "ethos-u"
                #     if func.attrs:
                #         print("函数属性 (attrs):")
                #         for key, value in func.attrs.items():
                #             print(f"  - {key}: {value}")
                #     else:
                #         print("函数属性 (attrs): 无")

                #     # 3. 打印函数体的前 100 个字符
                #     # 完整打印 func 会很长，和直接 print(mod) 类似
                #     # 这样可以只看个大概
                #     print(f"函数体预览: {str(func)}...")
                 
                # --------------------------------------
                
                # is_npu_func 是一个筛选器，它会检查函数的 attrs，找到那些被标记了 "Compiler": "ethos-u" 的函数，是那个很大的函数@tvm_default...
                npu_functions = filter(lambda x: is_npu_func(x[1]), mod.functions.items())
 
                i=0
                # 经理找到了所有 NPU 部门，开始逐一处理
                # global_var是@tvm_default...那个，但是这个很长的里面会出现：
                #  func还是那个大的
                for global_var, func in npu_functions:
                    i=i+1
                    print("i=",i)
                    print("=========create_npu_function_pass=============")
                    print("global_var:",global_var)
                    print("function:",func)
                    print("==========transform_npu_function start===========")

                    # 调用传入的pass：npu_pass_class
                    npu_pass = npu_pass_class(*self.args, **self.kwargs)
                    # ...然后调用专家的核心方法，把这一个部门交给他处理...
                    func = npu_pass.transform_npu_function(global_var, func) # 这个方法需要在 npu_pass 里面定义!
                    # 将专家优化后的部门方案更新回公司
                    mod.update_func(global_var, func)
                print("i=",i)
                return mod

        return ModulePassWrapper

    return decorator
