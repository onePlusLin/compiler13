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
# pylint: disable=unused-import
"""Common data structures across all IR variants.
# 该模块汇总了TVM中所有IR（中间表示）变体共用的核心数据结构，
# 为不同IR（如用于高层图优化的Relay IR、底层张量计算的TIR等）提供统一的基础组件，
# 避免重复定义，确保IR体系的模块化、兼容性和可扩展性，是TVM编译栈中IR操作的"基础工具箱"。
"""
# 诊断与插桩工具：用于IR相关的错误提示、调试和性能分析
from . import diagnostics, instrument, transform
# 代数数据类型（ADT）：用于定义复合数据类型（如枚举、结构体）
from .adt import Constructor, TypeData
# 仿射类型：支持带线性变换的类型描述（如动态张量形状）
from .affine_type import TensorAffineType, TupleAffineType
# 属性系统：为IR节点附加元信息（如算子参数、函数属性）
from .attrs import Attrs, DictAttrs, make_node
# 基础核心组件：IR节点基类、序列化、结构化比较等核心能力
from .base import (
    EnvFunc,          # 访问全局环境函数的封装
    Node,             # 所有IR节点的基类，提供结构化比较、哈希等能力
    SourceName,       # 记录IR节点对应的源码文件名
    Span,             # 记录IR节点在源码中的位置（行号、列号）
    SequentialSpan,   # 多个Span的组合（如跨多行的IR节点）
    assert_structural_equal,  # 断言两个IR节点结构化相等（语义等价）
    load_json,        # 从JSON反序列化IR节点
    save_json,        # 将IR节点序列化为JSON
    structural_equal, # 判断两个IR节点是否结构化相等
    structural_hash   # 计算IR节点的结构化哈希值（用于哈希表）
)
# 容器类型：IR专用的数组和映射，支持结构化比较
from .container import Array, Map
# 表达式相关：IR中计算逻辑的基础表示
from .expr import (
    BaseExpr,     # 所有表达式的基类（如常量、变量、算子调用）
    GlobalVar,    # 全局变量引用（如IR模块中的函数名、算子名）
    PrimExpr,     # 底层原始表达式（如整数、算术运算，用于TIR）
    Range,        # 范围描述（如循环索引范围 0<=i<10）
    RelayExpr     # 高层Relay IR表达式（如神经网络层调用，用于图优化）
)
# 函数相关：IR中函数的基础定义
from .function import (
    BaseFunc,     # 所有函数的基类（如Relay的Function、TIR的PrimFunc）
    CallingConv   # 函数调用约定（如CPU/GPU调用规则，确保跨设备兼容性）
)
# 内存池配置：用于IR优化中的内存管理（如常量/工作区内存分配策略）
from .memory_pools import (
    ConstantMemoryPools,  # 常量内存池集合
    ConstantPoolInfo,     # 常量内存池的配置信息
    PoolInfo,             # 内存池的基础信息
    PoolInfoProperties,   # 内存池的属性（如大小、对齐方式）
    WorkspaceMemoryPools, # 工作区内存池集合
    WorkspacePoolInfo     # 工作区内存池的配置信息
)
# IR模块：IR的顶层容器，包含函数、全局变量、类型定义等
from .module import IRModule
# 算子相关：算子定义与注册工具
from .op import (
    Op,                    # 算子基类（封装计算逻辑、输入输出类型等）
    register_intrin_lowering,  # 注册算子的底层实现（如为conv2d添加GPU优化）
    register_op_attr        # 注册算子的属性（如计算复杂度、支持的设备）
)
# 张量类型：描述张量的数据类型和维度
from .tensor_type import TensorType
# 类型系统：IR中数据类型的统一描述（支撑类型检查、推断）
from .type import (
    FuncType,        # 函数类型（描述输入输出类型）
    GlobalTypeVar,   # 全局类型变量（如模块中定义的自定义类型）
    IncompleteType,  # 未完全定义的类型（用于类型推断过程）
    PointerType,     # 指针类型（用于内存地址描述）
    PrimType,        # 原始类型（如int32、float32等基础类型）
    RelayRefType,    # 引用类型（如指针、对象引用，用于复杂数据结构）
    TupleType,       # 元组类型（多值打包，如函数返回多个结果）
    Type,            # 所有类型的基类
    TypeConstraint,  # 类型约束（如"T必须是数值类型"）
    TypeKind,        # 类型类别（如基础类型、复合类型）
    TypeVar          # 类型变量（如泛型函数中的类型参数T）
)
# 类型关系：描述类型间的关系（支撑类型检查中的关系验证）
from .type_relation import TypeCall, TypeRelation