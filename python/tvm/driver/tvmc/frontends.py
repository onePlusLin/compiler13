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
"""
Provides support to parse models from different frameworks into Relay networks.

Frontend classes do lazy-loading of modules on purpose, to reduce time spent on
loading the tool.
"""
import logging
import os
import sys
import re
import importlib
from abc import ABC
from abc import abstractmethod
from typing import Optional, List, Dict
from pathlib import Path

import numpy as np

from tvm import relay
from tvm import parser
from tvm.driver.tvmc import TVMCException, TVMCImportError
from tvm.driver.tvmc.model import TVMCModel


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


class Frontend(ABC):
    """Abstract class for command line driver frontend.

    Provide a unified way to import models (as files), and deal
    with any required preprocessing to create a TVM module from it."""

    @staticmethod
    @abstractmethod
    def name():
        """Frontend name"""

    @staticmethod
    @abstractmethod
    def suffixes():
        """File suffixes (extensions) used by this frontend"""

    @abstractmethod
    def load(self, path, shape_dict=None, **kwargs):
        """Load a model from a given path.

        Parameters
        ----------
        path: str
            Path to a file
        shape_dict: dict, optional
            Mapping from input names to their shapes.

        Returns
        -------
        mod : tvm.IRModule
            The produced relay module.
        params : dict
            The parameters (weights) for the relay module.

        """


def lazy_import(pkg_name, from_pkg_name=None, hide_stderr=False):
    """Lazy import a frontend package or subpackage"""
    try:
        return importlib.import_module(pkg_name, package=from_pkg_name)
    except ImportError as error:
        raise TVMCImportError(pkg_name) from error
    finally:
        if hide_stderr:
            sys.stderr = stderr


class KerasFrontend(Frontend):
    """Keras frontend for TVMC"""

    @staticmethod
    def name():
        return "keras"

    @staticmethod
    def suffixes():
        return ["h5"]

    def load(self, path, shape_dict=None, **kwargs):
        # pylint: disable=C0103
        tf = lazy_import("tensorflow")
        keras = lazy_import("keras", from_pkg_name="tensorflow")

        # tvm build currently imports keras directly instead of tensorflow.keras
        try:
            model = keras.models.load_model(path)
        except ValueError as err:
            raise TVMCException(str(err))

        # There are two flavours of keras model, sequential and
        # functional, TVM expects a functional model, so convert
        # if required:
        if self.is_sequential_p(model):
            model = self.sequential_to_functional(model)

        in_shapes = []
        for layer in model._input_layers:
            if tf.executing_eagerly():
                in_shapes.append(tuple(dim if dim is not None else 1 for dim in layer.input.shape))
            else:
                in_shapes.append(
                    tuple(dim.value if dim.value is not None else 1 for dim in layer.input.shape)
                )

        inputs = [np.random.uniform(size=shape, low=-1.0, high=1.0) for shape in in_shapes]
        input_shapes = {name: x.shape for (name, x) in zip(model.input_names, inputs)}
        if shape_dict is not None:
            input_shapes.update(shape_dict)
        kwargs.setdefault("layout", "NHWC")
        return relay.frontend.from_keras(model, input_shapes, **kwargs)

    def is_sequential_p(self, model):
        keras = lazy_import("keras", from_pkg_name="tensorflow")
        return isinstance(model, keras.models.Sequential)

    def sequential_to_functional(self, model):
        keras = lazy_import("keras", from_pkg_name="tensorflow")
        assert self.is_sequential_p(model)
        input_layer = keras.layers.Input(batch_shape=model.layers[0].input_shape)
        prev_layer = input_layer
        for layer in model.layers:
            prev_layer = layer(prev_layer)
        model = keras.models.Model([input_layer], [prev_layer])
        return model


class OnnxFrontend(Frontend):
    """ONNX frontend for TVMC"""

    @staticmethod
    def name():
        return "onnx"

    @staticmethod
    def suffixes():
        return ["onnx"]

    def load(self, path, shape_dict=None, **kwargs):
        onnx = lazy_import("onnx")

        # pylint: disable=E1101
        model = onnx.load(path)

        return relay.frontend.from_onnx(model, shape=shape_dict, **kwargs)


class TensorflowFrontend(Frontend):
    """TensorFlow frontend for TVMC"""

    @staticmethod
    def name():
        return "pb"

    @staticmethod
    def suffixes():
        return ["pb"]

    def load(self, path, shape_dict=None, **kwargs):
        tf = lazy_import("tensorflow")
        tf_testing = lazy_import("tvm.relay.testing.tf")

        with tf.io.gfile.GFile(path, "rb") as tf_graph:
            content = tf_graph.read()

        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(content)
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        logger.debug("parse TensorFlow model and convert into Relay computation graph")
        return relay.frontend.from_tensorflow(graph_def, shape=shape_dict, **kwargs)


# #############################  后续分配buffer的时候会出错！，详情见ytdebug5.log  ###############################################################
# import tvm
# from tvm.ir import structural_equal # <--- 关键：导入结构相等比较函数
# from tvm.relay.expr_functor import ExprMutator

# # 这是实现正确逻辑的核心类
# class QuantizedSquareRewriter(ExprMutator):
#     """
#     为重复的节点新建一个 add节点，使它加上“0”!
#     """
#     def visit_call(self, call):
#         # --- 阶段 1: 分析 ---
#         # 在原始的、类型完整的 `call` 节点上进行所有检查和信息提取。
        
#         is_target_op = False
        
#         # 预先提取信息，以防我们后面需要它们
#         input_dtype = None
#         input_shape = None
#         lhs_scale = None
#         lhs_zero_point = None

#         qnn_mul_op = relay.op.get("qnn.mul")
#         if isinstance(call.op, tvm.ir.Op) and call.op == qnn_mul_op:
#             # 确保参数足够多，避免索引错误
#             if len(call.args) == 8:
#                 lhs = call.args[0]
#                 rhs = call.args[1]
#                 if structural_equal(lhs, rhs):
#                     is_target_op = True
#                     # 从原始的、有类型的 lhs 节点缓存所有需要的信息
#                     input_dtype = lhs.checked_type.dtype
#                     input_shape = [int(x) for x in lhs.checked_type.shape]
#                     lhs_scale = call.args[2]
#                     lhs_zero_point = call.args[3]

#         # --- 阶段 2: 递归 ---
#         # 现在，安全地调用父类的 visit_call，会递归处理所有的输入节点（因为不是所要的mul)重新组装出一个 qnn.mul 节点，并将其作为 new_call 返回)
#         # 返回的 new_call 及其参数可能已经没有类型信息了。
#         new_call = super().visit_call(call)

#         # --- 阶段 3: 重构 ---
#         # 仅当我们之前标记了这是一个目标节点时，才进行重构。
#         if is_target_op:
#             print("V3: 成功匹配并重写 qnn.mul(x, x) 节点")
            
#             # 从可能无类型的 new_call 中获取被修改过的 lhs
#             mutated_lhs = new_call.args[0]

#             # 使用我们在阶段 1 缓存的信息来构建新节点
#             zero_data = relay.const(np.zeros(input_shape, dtype=input_dtype))
#             zero_scale = relay.const(1.0, "float32")
#             zero_zero_point = relay.const(0, "int32")

#             new_rhs = relay.qnn.op.add(
#                 lhs=mutated_lhs,
#                 rhs=zero_data,
#                 lhs_scale=lhs_scale,
#                 lhs_zero_point=lhs_zero_point,
#                 rhs_scale=zero_scale,
#                 rhs_zero_point=zero_zero_point,
#                 output_scale=lhs_scale,
#                 output_zero_point=lhs_zero_point,
#             )

#             new_args = list(new_call.args)
#             new_args[1] = new_rhs
#             new_args[4] = lhs_scale
#             new_args[5] = lhs_zero_point

#             return relay.Call(new_call.op, new_args, new_call.attrs)
        
#         # 如果不是目标节点，直接返回递归的结果
#         return new_call

#######################################################################

import tvm
from tvm.ir import structural_equal
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.contrib import ethosu
import numpy as np

class QuantizedSquareRewriter(ExprMutator):
    """
    V6: 最终解决方案。使用查找表 (LUT) 彻底替换 qnn.mul(x, x) 模式。
    这与 Ethos-U 等 NPU 后端的设计理念完全一致。
    """
    def visit_call(self, call):
        # --- 阶段 1: 分析 ---
        is_target_op = False
        # 预先提取所有需要的量化参数
        lhs_scale, lhs_zero_point = None, None
        output_scale, output_zero_point = None, None
        output_dtype = None

        qnn_mul_op = relay.op.get("qnn.mul")
        if isinstance(call.op, tvm.ir.Op) and call.op == qnn_mul_op:
            if len(call.args) == 8:
                lhs = call.args[0]
                rhs = call.args[1]
                if structural_equal(lhs, rhs):
                    is_target_op = True
                    # 从原始的、类型完整的节点缓存所有需要的信息
                    lhs_scale = call.args[2]
                    lhs_zero_point = call.args[3]
                    output_scale = call.args[6]
                    output_zero_point = call.args[7]
                    output_dtype = call.checked_type.dtype

        # --- 阶段 2: 递归 ---
        new_call = super().visit_call(call)

        # --- 阶段 3: 重构 ---
        if is_target_op:
            print("V9: 成功匹配 qnn.mul(x, x)，将使用 dequantize->power->quantize 降级到 CPU 执行。")
            
            mutated_lhs = new_call.args[0]

            # 1. 反量化: 将 int8 输入转为 float32
            dequantized_input = relay.qnn.op.dequantize(
                data=mutated_lhs,
                input_scale=lhs_scale,
                input_zero_point=lhs_zero_point
            )

            # 2. 计算浮点平方: 使用 relay.power
            # 指数必须是一个浮点常量
            exponent = relay.const(2.0, "float32")
            float_squared = relay.power(dequantized_input, exponent)

            # 3. 再量化: 将 float32 结果转回 int8
            requantized_output = relay.qnn.op.quantize(
                data=float_squared,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
                out_dtype=output_dtype
            )
            
            return requantized_output
        
        return new_call


# 这是提供给您的简单易用的API函数
def fix_quantized_square(mod: tvm.IRModule) -> tvm.IRModule:
    """
    遍历一个IRModule，查找所有 qnn.mul(x, x) 的实例，
    并用一个等价的、不使用该模式的计算序列替换它们。

    参数:
        mod: 原始的 TVM IRModule。

    返回:
        一个新的、经过修正的 TVM IRModule。
    """
    # 确保类型信息存在，这对于重写是必要的
    mod = relay.transform.InferType()(mod)
    print("--- 原始 IRModule ---")
    print(mod.astext(show_meta_data=False))
    
   # 实例化并应用重写器
    rewriter = QuantizedSquareRewriter()
    new_body = rewriter.visit(mod["main"].body) # mod["main"] == relay.Function，两部分：params、body--Expression
    
    # 创建一个新的函数和模块
    new_func = relay.Function(mod["main"].params, new_body) # mod["main"].params这是？
    new_mod = tvm.IRModule.from_expr(new_func)
    new_mod = relay.transform.InferType()(new_mod)

    print("\n--- 变换后的 IRModule ---")
    print(new_mod)
    # exit()
        
    return new_mod

############################################################################################





class TFLiteFrontend(Frontend):
    """TFLite frontend for TVMC"""

    @staticmethod
    def name():
        return "tflite"

    @staticmethod
    def suffixes():
        return ["tflite"]

    def load(self, path, shape_dict=None, **kwargs):
        # # 延迟导入 tflite.Model 模块。
        model = lazy_import("tflite.Model")

        # 以二进制读取模式（"rb"）打开文件。
        # 读取文件的全部二进制内容到 'content' 变量中
        with open(path, "rb") as tf_graph:
            content = tf_graph.read()

        # tflite.Model.Model is tflite.Model in 1.14 and 2.1.0
        try:
            # 尝试使用较新版本 tflite 库的 API 来解析 'content'
            tflite_model = model.Model.GetRootAsModel(content, 0)
        except AttributeError:
            tflite_model = model.GetRootAsModel(content, 0)

        try:
            version = tflite_model.Version()
            logger.debug("tflite version %s", version)
        except Exception:
            raise TVMCException("input file not tflite")

        if version != 3:
            raise TVMCException("input file not tflite version 3")

        logger.debug("parse TFLite model and convert into Relay computation graph")
        
        # print("tflite_model",tflite_model)tflite_model <tflite.Model.Model object at 0x7b3159e68af0>
        
        mod, params = relay.frontend.from_tflite(tflite_model, shape_dict=shape_dict, **kwargs)
        # print("最最最最最最最最最最最最开始的模型：",mod)
        # 在这里加入一个纠正吧：
        
        # fixed_mod = fix_quantized_square(mod)
    
        
        return mod ,params


class PyTorchFrontend(Frontend):
    """PyTorch frontend for TVMC"""

    @staticmethod
    def name():
        return "pytorch"

    @staticmethod
    def suffixes():
        # Torch Script is a zip file, but can be named pth
        return ["pth", "zip"]

    def load(self, path, shape_dict=None, **kwargs):
        torch = lazy_import("torch")

        if shape_dict is None:
            raise TVMCException("--input-shapes must be specified for %s" % self.name())

        traced_model = torch.jit.load(path)
        traced_model.eval()  # Switch to inference mode

        # Convert shape dictionary to list for Pytorch frontend compatibility
        input_shapes = list(shape_dict.items())

        logger.debug("parse Torch model and convert into Relay computation graph")
        return relay.frontend.from_pytorch(
            traced_model, input_shapes, keep_quantized_weight=True, **kwargs
        )


class PaddleFrontend(Frontend):
    """PaddlePaddle frontend for TVMC"""

    @staticmethod
    def name():
        return "paddle"

    @staticmethod
    def suffixes():
        return ["pdmodel"]

    def load(self, path, shape_dict=None, **kwargs):
        # pylint: disable=C0415
        import paddle

        paddle.enable_static()
        paddle.disable_signal_handler()

        if not os.path.exists(path):
            raise TVMCException("File {} is not exist.".format(path))
        if not path.endswith(".pdmodel"):
            raise TVMCException("Path of model file should be endwith suffixes '.pdmodel'.")
        prefix = "".join(path.strip().split(".")[:-1])
        params_file_path = prefix + ".pdiparams"
        if not os.path.exists(params_file_path):
            raise TVMCException("File {} is not exist.".format(params_file_path))

        # pylint: disable=E1101
        exe = paddle.static.Executor(paddle.CPUPlace())
        prog, _, _ = paddle.static.load_inference_model(prefix, exe)

        return relay.frontend.from_paddle(prog, shape_dict=shape_dict, **kwargs)


class RelayFrontend(Frontend):
    """Relay frontend for TVMC"""

    @staticmethod
    def name():
        return "relay"

    @staticmethod
    def suffixes():
        return ["relay"]

    def load(self, path, shape_dict=None, **kwargs):
        with open(path, "r", encoding="utf-8") as relay_text:
            text = relay_text.read()
        if shape_dict is None:
            logger.warning(
                "Specify --input-shapes to ensure that model inputs "
                "will not be considered as constants."
            )

        def _validate_text(text):
            """Check the provided file contents.
            The relay.txt artifact contained in the MLF is missing the version header and
            the metadata which is required to use meta[relay.Constant]."""

            if re.compile(r".*\#\[version\.*").match(text) is None:
                raise TVMCException(
                    "The relay model does not include the required version information."
                )
            if re.compile(r".*meta\[.+\].*", re.DOTALL).match(text):
                if "#[metadata]" not in text:
                    raise TVMCException(
                        "The relay model does not include the required #[metadata] section. "
                        "Use ir_mod.astext(show_meta_data=True) to export compatible code."
                    )

        _validate_text(text)

        ir_mod = parser.fromtext(text)

        if shape_dict:
            input_names = shape_dict.keys()
        else:
            input_names = []

        def _gen_params(ir_mod, skip_names=None):
            """Populate the all the params in the mode with ones."""
            main_func = ir_mod["main"]
            shape_dict = {p.name_hint: p.checked_type.concrete_shape for p in main_func.params}
            type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}
            params = {}
            for name, shape in shape_dict.items():
                if skip_names and name in skip_names:
                    continue

                if "int" in type_dict[name]:
                    data = np.random.randint(128, size=shape, dtype=type_dict[name])
                else:
                    data = np.random.uniform(-1, 1, size=shape).astype(type_dict[name])
                params[name] = data
            return params

        params = _gen_params(ir_mod, skip_names=input_names)

        return ir_mod, params


ALL_FRONTENDS = [
    KerasFrontend,
    OnnxFrontend,
    TensorflowFrontend,
    TFLiteFrontend,
    PyTorchFrontend,
    PaddleFrontend,
    RelayFrontend,
]


def get_frontend_names():
    """Return the names of all supported frontends

    Returns
    -------
    list : list of str
        A list of frontend names as strings

    """
    return [frontend.name() for frontend in ALL_FRONTENDS]


def get_frontend_by_name(name: str):
    """
    This function will try to get a frontend instance, based
    on the name provided.

    Parameters
    ----------
    name : str
        the name of a given frontend

    Returns
    -------
    frontend : tvm.driver.tvmc.Frontend
        An instance of the frontend that matches with
        the file extension provided in `path`.

    """

    for frontend in ALL_FRONTENDS:
        if name == frontend.name():
            return frontend()

    raise TVMCException(
        "unrecognized frontend '{0}'. Choose from: {1}".format(name, get_frontend_names())
    )


def guess_frontend(path: str):
    """
    This function will try to imply which framework is being used,
    based on the extension of the file provided in the path parameter.

    Parameters
    ----------
    path : str
        The path to the model file.

    Returns
    -------
    frontend : tvm.driver.tvmc.Frontend
        An instance of the frontend that matches with
        the file extension provided in `path`.

    """

    suffix = Path(path).suffix.lower()
    if suffix.startswith("."):
        suffix = suffix[1:]

    for frontend in ALL_FRONTENDS:
        if suffix in frontend.suffixes():
            return frontend()

    raise TVMCException("failed to infer the model format. Please specify --model-format")


# 加载模型并转换为 Relay
def load_model(
    path: str,
    model_format: Optional[str] = None,
    shape_dict: Optional[Dict[str, List[int]]] = None,
    **kwargs,
):
    """Load a model from a supported framework and convert it
    into an equivalent relay representation.

    Parameters
    ----------
    path : str
        The path to the model file.
    model_format : str, optional
        The underlying framework used to create the model.
        If not specified, this will be inferred from the file type.
    shape_dict : dict, optional
        Mapping from input names to their shapes.

    Returns
    -------
    tvmc_model : TVMCModel
        The produced model package.

    """

    if model_format is not None:
        frontend = get_frontend_by_name(model_format)
    else:
        frontend = guess_frontend(path)

    mod, params = frontend.load(path, shape_dict, **kwargs)

    return TVMCModel(mod, params)
