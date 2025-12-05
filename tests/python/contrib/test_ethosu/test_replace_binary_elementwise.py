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
import pytest

pytest.importorskip("ethosu.vela")

import tvm
from tvm import relay
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.ethosu.tir import spec
from tvm.relay.backend.contrib.ethosu.tir.compiler import _lower_to_tir
from .infra import make_ethosu_binary_elementwise, get_binary_elementwise_args


@pytest.mark.parametrize(
    "ifm_shape, ifm2_shape, ifm_channels, ifm2_channels, ifm_layout, ofm_layout, rounding_mode",
    [
        ((1, 5, 9, 3), (1, 5, 9, 3), 3, 3, "NHWC", "NHWC", "TFL"),
        ((1, 8, 3, 9, 16), (1, 8, 3, 9, 16), 40, 40, "NHCWB16", "NHCWB16", "NATURAL"),
        ((1, 8, 3, 9, 16), (1, 8, 3, 9, 16), 40, 40, "NHCWB16", "NHWC", "TRUNCATE"),
        ((1, 8, 9, 40), (1, 8, 9, 40), 40, 40, "NHWC", "NHCWB16", "TFL"),
        # Broadcast
        ((1, 5, 9, 3), (1, 1, 9, 1), 3, 1, "NHWC", "NHWC", "NATURAL"),
        ((1, 8, 9, 40), (1, 1, 1, 1), 40, 1, "NHWC", "NHCWB16", "TRUNCATE"),
    ],
)
@pytest.mark.parametrize("operator_type", ["ADD", "SUB", "MUL", "MIN", "MAX"])
@pytest.mark.parametrize("activation", ["NONE", "CLIP"])
def test_binary_elementwise_single(
    # 确保 TVM 能够正确地将各种二元元素级操作编译成适用于 Ethos-U 硬件的指令序列
    # 涵盖了实际使用中可能遇到的各种情况，保证了深度学习模型在该硬件上的正确性和性能
    # 这个pass就是判断在lower到tir之后的参数，步长是否符合硬件规格！
    ifm_shape,
    ifm2_shape,
    ifm_channels,
    ifm2_channels,
    ifm_layout,
    ofm_layout,
    rounding_mode,
    operator_type,
    activation,
):
    dtype = "int8"
    ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
    ifm2 = ifm # 测试完全一样输入的共享参数是否可以通过！
    # ifm2 = relay.var("ifm2", shape=ifm2_shape, dtype=dtype)
    # 创建一个二元元素级操作节点
    binary_elementwise = make_ethosu_binary_elementwise(
        ifm,
        ifm2,
        ifm_channels,
        ifm2_channels,
        operator_type,
        dtype,
        False,
        activation,
        ifm_layout,
        ifm_layout,
        ofm_layout,
        rounding_mode,
    )
    # 将自由变量变为函数的参数？
    # 自由变量。一般是外部定义，内部使用的，即free_vars 函数用于找出表达式中所有未被绑定的变量，ifm
    func = relay.Function(relay.analysis.free_vars(binary_elementwise), binary_elementwise)
    func = run_opt_pass(func, relay.transform.InferType())
    mod, _ = _lower_to_tir(func)
    
    print(mod["main"].script())
    
    # 遍历生成的 TIR 语句，提取二元操作的相关参数信息
    data = []
    def _visit(stmt):
        if isinstance(stmt, tvm.tir.Call):
            data.append(get_binary_elementwise_args(stmt))

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, _visit)
    # 计算步长
    if ifm_layout == "NHWC":
        ifm_stride_c = 1
        ifm_stride_w = ifm_shape[3] if ifm_shape[2] != 1 else 1
        ifm_stride_h = ifm_shape[2] * ifm_shape[3] if ifm_shape[1] != 1 else 1

        ifm2_stride_c = 1
        ifm2_stride_w = ifm2_shape[3] if ifm2_shape[2] != 1 else 1
        ifm2_stride_h = ifm2_shape[2] * ifm2_shape[3] if ifm2_shape[1] != 1 else 1

        ofm_height = ifm_shape[1]
        ofm_width = ifm_shape[2]
    else:
        ifm_stride_w = 16
        ifm_stride_c = 16 * ifm_shape[3]
        ifm_stride_h = 16 * ifm_shape[2] * ifm_shape[3]

        ifm2_stride_w = 16
        ifm2_stride_c = 16 * ifm2_shape[3]
        ifm2_stride_h = 16 * ifm2_shape[2] * ifm2_shape[3]

        ofm_height = ifm_shape[1]
        ofm_width = ifm_shape[3]

    if ofm_layout == "NHWC":
        ofm_stride_c = 1
        ofm_stride_w = ifm_channels if ofm_width > 1 else 1
        ofm_stride_h = ifm_channels * ofm_width if ofm_height > 1 else 1
    else:
        ofm_stride_w = 16
        ofm_stride_c = 16 * ofm_width
        ofm_stride_h = 16 * ofm_width * ((ifm_channels - 1) // 16 + 1)

    # 构建期望的序列化二元操作对象，包含了所有相关参数
    serial_binary_elementwise = spec.SerialBinaryElementwise(
        ifm=spec.SerialFeatureMap(
            data_type=dtype,
            height=ifm_shape[1],
            width=ifm_shape[2] if ifm_layout == "NHWC" else ifm_shape[3],
            channels=ifm_channels,
            tile_height_0=ifm_shape[1],
            tile_height_1=0,
            tile_width_0=ifm_shape[2] if ifm_layout == "NHWC" else ifm_shape[3],
            tile_address_0=0,
            tile_address_1=0,
            tile_address_2=0,
            tile_address_3=0,
            scale=1.0,
            zero_point=0,
            layout=ifm_layout,
            stride_h=ifm_stride_h,
            stride_w=ifm_stride_w,
            stride_c=ifm_stride_c,
        ),
        ifm2=spec.SerialFeatureMap(
            data_type=dtype,
            height=ifm2_shape[1],
            width=ifm2_shape[2] if ifm_layout == "NHWC" else ifm2_shape[3],
            channels=ifm2_channels,
            tile_height_0=ifm2_shape[1],
            tile_height_1=0,
            tile_width_0=ifm2_shape[2] if ifm_layout == "NHWC" else ifm2_shape[3],
            tile_address_0=0,
            tile_address_1=0,
            tile_address_2=0,
            tile_address_3=0,
            scale=1.0,
            zero_point=0,
            layout=ifm_layout,
            stride_h=ifm2_stride_h,
            stride_w=ifm2_stride_w,
            stride_c=ifm2_stride_c,
        ),
        ofm=spec.SerialFeatureMap(
            data_type=dtype,
            height=ofm_height,
            width=ofm_width,
            channels=ifm_channels,
            tile_height_0=ofm_height,
            tile_height_1=0,
            tile_width_0=ofm_width,
            tile_address_0=0,
            tile_address_1=0,
            tile_address_2=0,
            tile_address_3=0,
            scale=1.0,
            zero_point=0,
            layout=ofm_layout,
            stride_h=ofm_stride_h,
            stride_w=ofm_stride_w,
            stride_c=ofm_stride_c,
        ),
        operator_type=operator_type,
        reversed_operands=False,
        activation=spec.SerialActivation(
            op=activation,
            clip_min=10 if activation == "CLIP" else 0,
            clip_max=100 if activation == "CLIP" else 0,
        ),
        rounding_mode=rounding_mode,
        block_config=spec.SerialBlockConfig(0, 0, 0),
        rescale_config=spec.SerialRescaleConfig(False, 0, 0),
    )
    # 比较实际生成的结果与期望结果是否一致:
    assert data[0] == ["ethosu_binary_elementwise"] + list(serial_binary_elementwise)


@pytest.mark.parametrize(
    "ifm_shape, ifm2_shape, ifm_channels, ifm2_channels, ifm_layout, ofm_layout",
    [
        ((1, 5, 9, 3), (1, 5, 9, 3), 3, 3, "NHWC", "NHWC"),
        ((1, 8, 3, 9, 16), (1, 8, 3, 9, 16), 40, 40, "NHCWB16", "NHCWB16"),
        ((1, 8, 3, 9, 16), (1, 8, 3, 9, 16), 40, 40, "NHCWB16", "NHWC"),
        ((1, 8, 9, 40), (1, 8, 9, 40), 40, 40, "NHWC", "NHCWB16"),
        # Broadcast
        ((1, 5, 9, 3), (1, 1, 9, 1), 3, 1, "NHWC", "NHWC"),
        ((1, 8, 9, 40), (1, 1, 1, 1), 40, 1, "NHWC", "NHCWB16"),
    ],
)
@pytest.mark.parametrize("operator_type", ["SHR", "SHL"])
@pytest.mark.parametrize("rounding_mode", ["TFL", "NATURAL", "TRUNCATE"])
def test_shift_binary_elementwise_single(
    ifm_shape,
    ifm2_shape,
    ifm_channels,
    ifm2_channels,
    ifm_layout,
    ofm_layout,
    operator_type,
    rounding_mode,
):
    dtype = "int32"
    activation = "NONE"  # Only NONE is available if the activation type is int32
    ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
    ifm2 = relay.var("ifm2", shape=ifm2_shape, dtype=dtype)

    binary_elementwise = make_ethosu_binary_elementwise(
        ifm,
        ifm2,
        ifm_channels,
        ifm2_channels,
        operator_type,
        dtype,
        False,
        "NONE",
        ifm_layout,
        ifm_layout,
        ofm_layout,
        rounding_mode,
    )
    func = relay.Function(relay.analysis.free_vars(binary_elementwise), binary_elementwise)
    func = run_opt_pass(func, relay.transform.InferType())
    mod, _ = _lower_to_tir(func)
    data = []

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.Call):
            data.append(get_binary_elementwise_args(stmt))

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, _visit)
    if ifm_layout == "NHWC":
        ifm_stride_c = 1
        ifm_stride_w = ifm_shape[3] if ifm_shape[2] != 1 else 1
        ifm_stride_h = ifm_shape[2] * ifm_shape[3] if ifm_shape[1] != 1 else 1

        ifm2_stride_c = 1
        ifm2_stride_w = ifm2_shape[3] if ifm2_shape[2] != 1 else 1
        ifm2_stride_h = ifm2_shape[2] * ifm2_shape[3] if ifm2_shape[1] != 1 else 1

        ofm_height = ifm_shape[1]
        ofm_width = ifm_shape[2]
    else:
        ifm_stride_w = 16
        ifm_stride_c = 16 * ifm_shape[3]
        ifm_stride_h = 16 * ifm_shape[2] * ifm_shape[3]

        ifm2_stride_w = 16
        ifm2_stride_c = 16 * ifm2_shape[3]
        ifm2_stride_h = 16 * ifm2_shape[2] * ifm2_shape[3]

        ofm_height = ifm_shape[1]
        ofm_width = ifm_shape[3]

    if ofm_layout == "NHWC":
        ofm_stride_c = 1
        ofm_stride_w = ifm_channels if ofm_width > 1 else 1
        ofm_stride_h = ifm_channels * ofm_width if ofm_height > 1 else 1
    else:
        ofm_stride_w = 16
        ofm_stride_c = 16 * ofm_width
        ofm_stride_h = 16 * ofm_width * ((ifm_channels - 1) // 16 + 1)

    serial_binary_elementwise = spec.SerialBinaryElementwise(
        ifm=spec.SerialFeatureMap(
            data_type=dtype,
            height=ifm_shape[1],
            width=ifm_shape[2] if ifm_layout == "NHWC" else ifm_shape[3],
            channels=ifm_channels,
            tile_height_0=ifm_shape[1],
            tile_height_1=0,
            tile_width_0=ifm_shape[2] if ifm_layout == "NHWC" else ifm_shape[3],
            tile_address_0=0,
            tile_address_1=0,
            tile_address_2=0,
            tile_address_3=0,
            scale=1.0,
            zero_point=0,
            layout=ifm_layout,
            stride_h=ifm_stride_h,
            stride_w=ifm_stride_w,
            stride_c=ifm_stride_c,
        ),
        ifm2=spec.SerialFeatureMap(
            data_type=dtype,
            height=ifm2_shape[1],
            width=ifm2_shape[2] if ifm_layout == "NHWC" else ifm2_shape[3],
            channels=ifm2_channels,
            tile_height_0=ifm2_shape[1],
            tile_height_1=0,
            tile_width_0=ifm2_shape[2] if ifm_layout == "NHWC" else ifm2_shape[3],
            tile_address_0=0,
            tile_address_1=0,
            tile_address_2=0,
            tile_address_3=0,
            scale=1.0,
            zero_point=0,
            layout=ifm_layout,
            stride_h=ifm2_stride_h,
            stride_w=ifm2_stride_w,
            stride_c=ifm2_stride_c,
        ),
        ofm=spec.SerialFeatureMap(
            data_type=dtype,
            height=ofm_height,
            width=ofm_width,
            channels=ifm_channels,
            tile_height_0=ofm_height,
            tile_height_1=0,
            tile_width_0=ofm_width,
            tile_address_0=0,
            tile_address_1=0,
            tile_address_2=0,
            tile_address_3=0,
            scale=1.0,
            zero_point=0,
            layout=ofm_layout,
            stride_h=ofm_stride_h,
            stride_w=ofm_stride_w,
            stride_c=ofm_stride_c,
        ),
        operator_type=operator_type,
        reversed_operands=False,
        activation=spec.SerialActivation(
            op=activation,
            clip_min=0,
            clip_max=0,
        ),
        rounding_mode=rounding_mode,
        block_config=spec.SerialBlockConfig(0, 0, 0),
        rescale_config=spec.SerialRescaleConfig(False, 0, 0),
    )

    assert data[0] == ["ethosu_binary_elementwise"] + list(serial_binary_elementwise)


if __name__ == "__main__":
    tvm.testing.main()
