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
'''
负责将 Relay 函数 lowering 为 TIR（Tensor Intermediate Representation），
并针对 Ethos-U 的硬件架构（如计算单元、内存层次）生成初步的 TIR 代码。
'''
"""Codegen for Arm(R) Ethos(TM)-U NPU"""
from collections import defaultdict
from typing import List, Callable

from ethosu.vela import api as vapi
import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.tir.compiler import LowerToTIR
from tvm.relay.backend.contrib.ethosu.tir.scheduler import copy_constants
from tvm.contrib.ethosu.cascader import (
    cascade,            # 主要的调度函数，实现"级联"技术
    EthosuDeviceConfig, # 存储Ethos-U设备的配置信息
    CascaderOptions,    # 级联调度器的配置选项
    MemoryRegion,       # 表示内存区域信息
    extract_memory_info,# 从内存池中提取内存信息的工具函数
)
from tvm.relay.backend.contrib.ethosu.legalize import LegalizeEthosU
from tvm.relay.backend.contrib.ethosu import tir_to_cs_translator, util, vela_api
from tvm.relay.expr_functor import ExprMutator, ExprVisitor

# pylint: disable=unused-import
from tvm.relay.backend.contrib.ethosu.op import op_attrs
from tvm.relay.backend.contrib.ethosu import op

from . import _ffi_api

# 爷爷是Relay 中用于遍历和处理表达式（Expr）的抽象基类ExprFunctor 
# Expr（Relay 的表达式节点，如 Call、Function 等）
class OptimizeLUTs(ExprMutator):
    """A pass to merge an identity operator with a LUT based activation function with
    a preceding operator provided that operator can do a table lookup for the activation
    in the hardware
    这个pass可以合并一个带有基于LUT的激活函数的identity操作符与前面的操作符，前提是该操作符可以在硬件中进行激活的查找表。
    identity operator 是什么？ 一个占位符吧  contrib.ethosu.identity
    Arm Ethos-U NPU 的硬件设计决定了，激活函数（如 ReLU, Tanh, Sigmoid）不能独立执行。它必须作为另一个主操作（如卷积、加法、池化）的一个附加步骤，在硬件层面被融合执行。
    """

    def __init__(self):
        super().__init__()
        self.lut_ops = {
            "contrib.ethosu.conv2d": op.ethosu_conv2d,
            "contrib.ethosu.depthwise_conv2d": op.ethosu_depthwise_conv2d,
            "contrib.ethosu.pooling": op.ethosu_pooling,
            "contrib.ethosu.binary_elementwise": op.ethosu_binary_elementwise,
        }

    def create_op_with_lut(self, call):
        """
        将contrib.ethosu.identity的op（第一个参数-上游节点）和携带的LUTtiqu1,创造一个op，到列表尾
        Extract the parameters and attributes from the NPU operator and create
        a new operator with LUT.

        Parameters
        ----------
        call : tvm.relay.expr.Call
            The current call node being visited.

        Returns
        -------
        tvm.relay.expr.Call
            The new operator with LUT.
        """
        identity = call
 
        ethosu_op = call.args[0]
        lut = identity.args[1] #  LUT 数据 (查找表参数)
        
        activation = identity.attrs.activation

        new_attrs = dict(ethosu_op.attrs)
        new_attrs["activation"] = activation

        # Assume that LUT is always the last argument
        new_args = ethosu_op.args[:-1] + [lut]
        assert ethosu_op.op.name in self.lut_ops.keys()

        # 构造新的算子调用
        return self.lut_ops[ethosu_op.op.name](*new_args, **new_attrs)


# func :<class 'tvm.relay.function.Function'> func在遍历整个图的时候好像会将图转为一个expr,从而可以访问callnode，在继承的visit中转换
    def visit_call(self, call: tvm.relay.expr.Call) -> tvm.relay.expr.Call:
        """Recursively visit call nodes in the input graph and if an ethosu.identity
        operator with LUT is found and the preceding operator has a LUT attribute, create
        a new NPU operator.
        
        递归地访问输入图中的调用节点，如果找到带有查找表（LUT）的 ethosu.identity 操作符且其前一个操作符是（lut_ops），本call有激活属性
        ，则创建一个新的 NPU 操作符。
        identity 的核心作用，是作为一个软件层面的占位符（Placeholder），来表示一个独立的激活函数。

        Parameters
        ----------
        call : tvm.relay.expr.Call
            The current call node being visited.

        Returns
        -------
        tvm.relay.expr.Call
            The input call node in the case the current call node does
            not refer to an Op. Else, a new call node with a new operator.
        """
        new_call = call
        lut_activations = ["TANH", "LUT", "SIGMOID"]
        
        if isinstance(call.op, tvm.ir.Op) and isinstance(call.args[0], tvm.relay.expr.Call):

            producer_op = call.args[0] # 生产者节点，输入的第一个节点

            # Check if the producer can do a LUT operation
            # 如果这个call节点的上游节点符合条件，当前call节点是不是identity，该算子的属性（配置参数）activation是LUT激活函数，则可以被创造一个LUTop
            if (
                producer_op.op.name in self.lut_ops.keys()     # bug for yolov5s.tflite
                and call.op.name == "contrib.ethosu.identity"
                and call.attrs.activation in lut_activations
            ):
                # Check the producer doesn't already have a LUT
                has_lut = producer_op.attrs.activation in lut_activations
                if not has_lut:
                    new_call = self.create_op_with_lut(call)

        # 递归处理输入节点
        new_call = super().visit_call(new_call)

        return new_call

# 装饰器的作用是将这个类注册为一个 Relay pass，从而可以直接使用优化（需要实现transform_npu_function函数）
# 这个装饰器会找到所有被标记为ethosu的函数，然后调用transform_npu_function,所以fn不是全部的mod，而是全部npn函数
@util.create_npu_function_pass(opt_level=1)
class LUTsOptimizer:
    """Register LUTsOptimizer as a relay pass.
        优化和合并专用于NPU的LUTs
    """

    def transform_npu_function(self, _, func: relay.Function) -> relay.Function:
        """Visit relay nodes in the given NPU function.

        Parameters
        ----------
        func : tvm.relay.function.Function
            The function to apply the optimization pass for multiple LUTs to.

        Returns
        -------
        mod : tvm.IRModule
            New module with optimized LUTs.
        """
        print("ZEdebug:class LUTsOptimizer:ir-codegen.py, func type:\n",type(func))
        print("func 是整一个model吗？:\n",func) #在outline之后，npu函数会被分离出来，一般就是一个计算图！
        return OptimizeLUTs().visit(func) 

    # 使得类的实例可以像函数一样被调用
    def __call__(self, *args, **kwargs):
        pass


class AnalyzeConsumers(ExprVisitor):
    """Traverses the graph to determine consumers that are NPU operations and
    which have restrictions to use NHCWB16 layout. The result is maintained in
    `npu_consumers` and `restrictions`.
    编译图确定哪些消费者是NPU op，并检查哪些被限制转为NHCW

    Attributes
    ----------
    npu_consumers : Dict[tvm.relay.expr.Call, List[bool]]
        Mapping from NPU operation to list of boolean values that represent
        whether or not each consumer is an NPU operation.
    restrictions : Dict[tvm.relay.expr.Call, List[bool]]
        Mapping from NPU operation to list of boolean values that represent
        whether or not operation has restrictions to use NHCWB16 layout.
    optimize_ops : Dict[str, Callable]
        A map from NPU operation name to function that creates NPU operation.
    """

    def __init__(self, optimize_ops):
        self.npu_consumers = defaultdict(list)
        self.restrictions = defaultdict(list)
        self.optimize_ops = optimize_ops
        super().__init__()

    def visit_call(self, call: relay.Call):
        is_npu_consumer = call.op.name in self.optimize_ops
        args = []

        # Expand tuples
        for arg in call.args:
            if isinstance(arg, relay.Tuple):
                args.extend(arg.fields)
            else:
                args.append(arg)

        for arg in args:
            if isinstance(arg, relay.Call) and arg.op.name in self.optimize_ops:
                self.npu_consumers[arg].append(is_npu_consumer)
                # ReduceSum requires NHWC input in case input tensor has type int32 or
                # accelerator is Ethos_U65_512
                # https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela/+/refs/tags/3.7.0/ethosu/vela/graph_optimiser_util.py#126
                has_restrictions = (
                    call.op.name == "contrib.ethosu.pooling"
                    and call.attrs["pooling_type"] == "SUM"
                    and (
                        arg.checked_type.dtype == "int32"
                        or vela_api.get_accelerator_config() == vapi.NpuAccelerator.Ethos_U65_512
                    )
                )
                self.restrictions[arg].append(has_restrictions)

        super().visit_call(call)


class LayoutOptimization(ExprMutator):
    """A pass to optimize the layout of NPU operations by converting to brick format (NHCWB16).
    This pass traverses the graph and attempts to alter the input/output layouts when an NPU
    operation is visited. Whether or not the input/output layout can be altered for a given NPU
    operation depends on the following:

    Check alter input layout: For each argument, if the producer is also an NPU operation and
        its output is altered to brick format and there are no restrictions, then the input layout
        with respect to the current argument is altered to brick format.

    Check alter output layout: If all consumers (child nodes) are an NPU operation and
        there are no restrictions, then the output layout is altered to brick format.

    Note
    ----
    In order for this pass to be run, the consumers of each NPU operation must first be analyzed
    by the `AnalyzeConsumers` pass, since Relay doesn't keep a reference to child nodes.

    Attributes
    ----------
    npu_consumers : Dict[tvm.relay.expr.Call, List[bool]]
        A map from current call to a list boolean values that state whether or not each consumer
        is an NPU operation.
    restrictions : Dict[tvm.relay.expr.Call, List[bool]]
        A map from current call to a list boolean values that state
        whether or not operation has restrictions to use NHCWB16 layout.
    optimize_ops : Dict[str, Callable]
        A map from NPU operation name to function that creates NPU operation.
    """

    def __init__(self, npu_consumers, restrictions, optimize_ops):
        self.npu_consumers = npu_consumers
        self.restrictions = restrictions
        self.optimize_ops = optimize_ops
        super().__init__()

    def alter_ethosu_op_layout(self, call: tvm.relay.expr.Call) -> tvm.relay.expr.Call:
        """Alter the layouts of given NPU operation to brick format if possible.

        Parameters
        ----------
        call : tvm.relay.expr.Call
            The call pointing to an NPU operation that will be checked if
            the layout needs altering.

        Returns
        -------
        new_call : tvm.relay.expr.Call
            New call with altered layouts.
        """

        def are_all_consumers_npu(call):
            """
            检查一个节点的消费者节点是不是全是
            Check whether or not each consumer is an NPU operation.
            Parameters
            ----------
            call : tvm.relay.expr.Call
                The call pointing to an NPU operation.

            Returns
            -------
            all_consumers_npu : bool
                Whether each consumer is an NPU operation.
            """
            consumers = self.npu_consumers[call]
            return consumers and all(consumers)

        def check_restrictions(call):
            """
            Check if there are any restrictions for call to use NHCWB16 layout.
            Parameters
            ----------
            call : tvm.relay.expr.Call
                The call pointing to an NPU operation.

            Returns
            -------
            any_restrictions : bool
                Whether there are restrictions.
            """
            restrictions = self.restrictions[call]
            return restrictions and any(restrictions)

        assert isinstance(call.attrs, tvm.ir.Attrs), (
            f"The attributes for operator '{call.op.name}' could not be "
            "found. Did you register the relay.attrs.Ethosu<opname>Attrs "
            "object in python api?"
        )

        new_attrs = dict(call.attrs)

        # Check if we can rewrite the input layouts
        input_count = 0
        for arg in call.args:
            input_count += 1
            if arg not in self.npu_consumers:
                continue
            parent_has_brick_output = are_all_consumers_npu(arg) # 父母就是指上游即输入吧？
            parent_has_restrictions = check_restrictions(arg)
            if parent_has_brick_output and not parent_has_restrictions:
                 # 父节点已经将输出优化为 NHCWB16，所以当前节点可以接受这种布局
                layout_string = "ifm_layout" if input_count <= 1 else f"ifm{input_count}_layout"
                new_attrs[layout_string] = "NHCWB16"

        # Check if we can rewrite the output layouts
        has_brick_output = are_all_consumers_npu(call)
        has_restrictions = check_restrictions(call)
        if has_brick_output and not has_restrictions:
            new_attrs["ofm_layout"] = "NHCWB16"

        name = call.op.name
        return self.optimize_ops[name](*call.args, **new_attrs)

    def visit_call(self, call: tvm.relay.expr.Call) -> tvm.relay.expr.Call:
        """Recursively visit call nodes in the input graph and alter the
        layout of an op if needed.

        Parameters
        ----------
        call : tvm.relay.expr.Call
            The current call node being visited.

        Returns
        -------
        tvm.relay.expr.Call
            The input call node in the case the current call node does
            not refer to an Op. Else, a new call node with altered Op
            attributes.
        """
        if isinstance(call.op, tvm.ir.Op) and call.op.name in self.optimize_ops:
            call = self.alter_ethosu_op_layout(call)
        return super().visit_call(call)


@util.create_npu_function_pass(opt_level=1)
class LayoutOptimizer:
    """Register LayoutOptimizer as a Relay pass."""

    def transform_npu_function(self, _, func: relay.Function) -> relay.Function:
        """A pass to optimize the layout of NPU operations. If both the
        producer and consumer of a tensor are NPU operators, then the
        layout is converted from NHWC to NHCWB16 as this is the layout NPU
        uses internally."""

        optimize_ops = {
            "contrib.ethosu.conv2d": op.ethosu_conv2d,
            "contrib.ethosu.depthwise_conv2d": op.ethosu_depthwise_conv2d,
            "contrib.ethosu.pooling": op.ethosu_pooling,
            "contrib.ethosu.binary_elementwise": op.ethosu_binary_elementwise,
            "contrib.ethosu.unary_elementwise": op.ethosu_unary_elementwise,
        }

        analyze = AnalyzeConsumers(optimize_ops)
        analyze.visit(func)
        return LayoutOptimization(analyze.npu_consumers, analyze.restrictions, optimize_ops).visit(
            func
        )

    def __call__(self, *args, **kwargs):
        pass


def IdentityOptimizer():  # pylint: disable=invalid-name
    """Pass that removes redundant identities

    Return
    ------
    Pass
        The module pass.
    """
    return _ffi_api.IdentityOptimizer()


def OutlineCompilerFunctions(compiler_name):  # pylint: disable=invalid-name
    """
    描绘出给定名字的编译器的函数的pass
    Pass that outlines functions given a named Compiler attribute.

    Parameters
    ----------
    compiler_name
        The name of the compiler to look for and outline.

    Return
    ------
    Pass
        The module pass.
    """
    return _ffi_api.OutlineCompilerFunctions(compiler_name)# 调用C++后端 "relay._transform.OutlineCompilerFunctions"


@tvm._ffi.register_func("relay.ext.ethos-u.constant_updater")
def constant_updater(expr, symbol):  # pylint: disable=unused-argument
    """
    The constant updater process happen after lowering in the core compiler.
    For the NPU, we dont want the build process to extract constants to be loaded in
    the runtime as we are embedding them inside the C runtime.Module.
    """
    return dict()


def _create_cascader(
    options: CascaderOptions,
    io_region: MemoryRegion,
    constant_region: MemoryRegion,
    working_regions: List[MemoryRegion],
    device_config: EthosuDeviceConfig,
) -> Callable:
    def _cascader(te_graph, const_dict, sch):
        cascade(
            sch,
            te_graph,
            const_dict,
            options,
            io_region,
            constant_region,
            working_regions,
            device_config,
        )

    return _cascader


def _ethos_u55_cascader(sram, enable_striping) -> Callable:
    # TODO(ekalda): Extract the flash info from ConstantPools once it is implemented
    flash = MemoryRegion(name="FLASH", size=10**7, read_bandwidth=4, write_bandwidth=4)

    device_config = EthosuDeviceConfig(util.get_accelerator_config())
    cascader_options = CascaderOptions(
        cascade_region=sram,
        max_proposals=64,
        stripe_factors=5,
        max_plan_size=10,
        always_copy_size=1024,
        max_open_plans=8,
        max_closed_plans=32,
        enable_striping=enable_striping,
    )
    return _create_cascader(
        options=cascader_options,
        io_region=sram,
        constant_region=flash,
        working_regions=[sram],
        device_config=device_config,
    )


def _calculate_memory_pressure(mod: tvm.ir.IRModule) -> int:
    """
    Calculates a worst-case estimate of the memory consumed at the callsite of
    each microNPU function. This value can be used as a hint to guide the cascader,
    indicating how aggressively it will need to optimize the input module to fit
    into the memory that remains in the memory workspace.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The input module

    Returns
    -------
    int
        Memory pressure value for the module.
    """
    memory_pressure = 0

    @util.create_npu_function_pass(opt_level=1)
    class CalculateMemoryPressure:
        """
        Traverse the module and get total memory used by external NPU functions.
        """

        def transform_npu_function(self, _, func: relay.Function) -> relay.Function:
            nonlocal memory_pressure
            max_val = max(func.attrs["used_memory"])
            memory_pressure += max_val
            return func

    CalculateMemoryPressure()(mod)  # pylint: disable=not-callable

    io_used_memory = 0
    if not tvm.tir.usmp.utils.use_workspace_io_is_enabled():
        io_used_memory = int(mod["main"].attrs["io_used_memory"])

    return memory_pressure - io_used_memory


@tvm._ffi.register_func("relay.ext.ethos-u.relay_to_tir")
def relay_to_tir(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:# 处理外部函数-external functions
    """
    This is the hook for python-based lowering of a Relay module which lowers NPU
    external functions to TIR.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        This is the Relay module.

    Returns
    -------
    mod : tvm.ir.IRModule
        The Relay module with scheduled NPU external functions.
    """
    
    
    print("ZEdebug:fun-relay_to_tir-codegen.py,!!! before !!! mod:\n")
    print(mod)
    mod = OutlineCompilerFunctions("ethos-u")(mod)
    '''
    工厂模式：
    OutlineCompilerFunctions("ethos-u") 返回一个 pass 对象
    然后这个 pass 对象被调用，传入 mod
    '''
    print("ZEdebug:fun-relay_to_tir-codegen.py,!!! after OutlineCompilerFunctions !!! mod:\n")
    print(mod)
    
    print("--------------LegalizeEthosU start---------------") 
    mod = LegalizeEthosU()(mod)
    print("ZEdebug:fun-relay_to_tir-codegen.py,!!! after LegalizeEthosU !!! mod:\n")
    print(mod)
    print("mod type:\n",type(mod))
    
    
    mod = LUTsOptimizer()(mod)      # bug yolov5s-int8.tflite
    print("ZEdebug:fun-relay_to_tir-codegen.py,!!! after LUTsOptimizer !!! mod:\n")
    print(mod)
    
    
    mod = relay.transform.InferType()(mod)
    print("ZEdebug:fun-relay_to_tir-codegen.py,!!! after InferType !!! mod:\n")
    print(mod)
    
    mod = IdentityOptimizer()(mod)
    print("ZEdebug:fun-relay_to_tir-codegen.py,!!! after IdentityOptimizer !!! mod:\n")
    print(mod)
    
    mod = LayoutOptimizer()(mod)
    print("ZEdebug:fun-relay_to_tir-codegen.py,!!! after LayoutOptimizer !!! mod:\n")
    print(mod)
    
    mod = relay.transform.InferType()(mod)
    print("ZEdebug:fun-relay_to_tir-codegen.py,!!! after InferType !!! mod:\n")
    print(mod)

    device_contexts = {
        gv: "ethos-u" for gv, _ in filter(lambda x: util.is_npu_func(x[1]), mod.functions.items())
    }
    mod = mod.with_attr("device_contexts", device_contexts)

    # Use the cascader if it is enabled for the U55 accelerator, otherwise use copy_constants
    # scheduler
    if util.is_cascader_enabled():
        if util.get_accelerator_config() == "ethos-u65-256":
            raise ValueError("Cascading is not supported for the U65 accelerator")

        workspace_memory_pools = mod.attrs["workspace_memory_pools"]

        if not workspace_memory_pools:
            raise ValueError("Workspace memory pool needs to be provided for the U55 cascader")
        if len(workspace_memory_pools.pools) != 1:
            raise ValueError("Exactly one workspace pool needs to be provided for the U55 cascader")

        memory_pressure = _calculate_memory_pressure(mod)
        sram = extract_memory_info(workspace_memory_pools.pools[0], memory_pressure)
        tir_mod = LowerToTIR(_ethos_u55_cascader(sram, util.is_striping_enabled()))(mod)
    else:
        scheduler = None if util.is_copying_constants_disabled() else copy_constants() # copy_constants
        tir_mod = LowerToTIR(scheduler)(mod) 
        
    print("ZEdebug:fun-relay_to_tir-codegen.py,!!! after !!!tir_mod:\n")
    print(tir_mod)

    return tir_mod


@tvm._ffi.register_func("relay.ext.ethos-u.primfunc_to_artifact")
def primfunc_to_artifact(primfunc: tvm.tir.PrimFunc) -> util.CompilationArtifact:
    """
    This is the hook for python-based lowering of TIR PrimFunc
    that has undergone unified optimization to compilation
    artifact destined for the microNPU.

    Parameters
    ----------
    primfunc : tir.PrimFunc
        TIR PrimFunc that has undergone unified optimizations

    Returns
    -------
    CompilationArtifact
        This is a structure that holds the binary artifacts
        for the microNPU
    """
    print("ZEdebug:fun-primfunc_to_artifact-codegen.py,!!! before !!! primfunc:\n",primfunc)
    # 提取函数符号和常量
    symbol = str(primfunc.attrs["global_symbol"])       # "global_symbol": "tvmgen_default_tvmgen_default_ethos_u_main_0"
    const_dict = primfunc.attrs["ethos-u.constants"]    # attr = {"ethos-u.constants": meta[Map][0],,,属性里面的常量，使用的是runtime.NDArray存储的
    
    # 创建一个新的TIR模块，并将PrimFunc添加到模块中
    tir_mod = tvm.IRModule()
    tir_mod[symbol] = primfunc
    
    print("ZEdebug:fun-primfunc_to_artifact-codegen.py,!!! before !!! tir_mod:\n",tir_mod)

    # 将常量字典中的数据从TVM的NDArray格式转换为NumPy数组格式
    const_dict_np = dict()
    for buffer_var in const_dict.keys():
        const_dict_np[buffer_var] = const_dict[buffer_var].numpy()

    # 将 tir_mod 转换成 cms（command stream）
    # cmms: 命令流（command stream）的十六进制字符串
    # encoded_constants: 编码后的常量数据（权重、偏置等）
    # base_addresses: 基地址信息列表
    cmms, encoded_constants, base_addresses = tir_to_cs_translator.translate(tir_mod, const_dict_np)
    return util.CompilationArtifact(symbol, cmms, encoded_constants, base_addresses)
