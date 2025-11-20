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
# pylint: disable=wildcard-import, redefined-builtin, invalid-name
"""The Relay IR namespace containing the IR definition and compiler."""
# 导入系统模块：os用于文件操作，sys.setrecursionlimit用于设置递归深度限制
import os
from sys import setrecursionlimit

# 导入Relay IR核心组件模块
from . import base                  # 基础IR定义（如位置信息、辅助工具）
from . import ty                    # 类型系统定义（如张量类型、函数类型等）
from . import expr                  # 表达式定义（如变量、调用、条件等）
from . import function              # 函数定义相关
from . import type_functor          # 类型相关的 functor（用于类型遍历/转换）
from . import expr_functor          # 表达式相关的 functor（用于表达式遍历/转换）
from . import adt                   # 代数数据类型（Algebraic Data Type）支持
from . import prelude               # Relay内置的基础工具函数集
from . import loops                 # 循环结构相关定义
from . import scope_builder         # 作用域构建工具（用于构建IR时管理变量作用域）
from .base import pretty_print, astext  # 从base模块导出格式化打印函数

# 导入编译和优化相关模块
from . import transform             # 编译转换 passes（如优化、分析等）
from . import analysis              # 静态分析工具（如依赖分析、类型检查等）
from . import collage               # 代码拼接/组合相关功能
from .build_module import build, create_executor, optimize  # 模块构建和执行相关函数
from .transform import build_config  # 构建配置工具
from . import debug                 # 调试工具
from . import param_dict            # 参数字典（模型参数序列化/反序列化）
from .backend import vm             # 虚拟机后端相关

# 导入核心算子（Root operators）
from .op import nn                  # 神经网络相关算子（如卷积、全连接等）
from .op import image               # 图像处理算子
from .op import annotation          # 注解相关算子
from .op import vision              # 计算机视觉专用算子
from .op import contrib             # 第三方贡献的算子
from .op import dyn                 # 动态形状相关算子
from .op import random              # 随机数生成算子
from .op.reduce import *            # 导入所有归约算子（如sum、mean等）
from .op.tensor import *            # 导入所有张量操作算子（如reshape、transpose等）
from .op.transform import *         # 导入所有转换算子
from .op.algorithm import *         # 导入算法相关算子

# 导入前端、后端和量化相关模块
from . import frontend              # 前端转换（如从PyTorch/TensorFlow导入模型）
from . import backend               # 后端代码生成（如CPU、GPU、专用硬件）
from . import quantize              # 模型量化工具
from . import data_dep_optimization # 数据依赖优化

# 导入方言（Dialects）- 这里是量化神经网络方言
from . import qnn                   # 量化神经网络（Quantized Neural Network）相关定义

# 从scope_builder模块导出ScopeBuilder类（用于构建作用域）
from .scope_builder import ScopeBuilder

# 导入内存规划相关的passes
from .transform import memory_plan  # 内存规划优化（如内存复用、布局优化等）

# 导入解析器相关功能（将文本转换为Relay IR）
from .parser import parse, parse_expr, fromtext, SpanCheck  # 解析函数和检查工具

# 设置递归深度限制（Relay IR可能包含深层嵌套结构，需要足够的递归深度）
setrecursionlimit(10000)

# 导出Span相关类（用于记录IR节点在源代码中的位置，便于调试）
Span = base.Span
SequentialSpan = base.SequentialSpan
SourceName = base.SourceName

# 导出类型系统相关类
Type = ty.Type                     # 所有类型的基类
TupleType = ty.TupleType           # 元组类型
TensorType = ty.TensorType         # 张量类型
TypeKind = ty.TypeKind             # 类型种类（如Int、Float等）
TypeVar = ty.TypeVar               # 类型变量（用于泛型）
ShapeVar = ty.ShapeVar             # 形状变量（用于动态形状）
TypeConstraint = ty.TypeConstraint # 类型约束
FuncType = ty.FuncType             # 函数类型
TypeRelation = ty.TypeRelation     # 类型关系（用于约束检查）
IncompleteType = ty.IncompleteType # 未完成类型（临时类型，用于类型推断）
scalar_type = ty.scalar_type       # 标量类型构造函数（如int32、float32）
RefType = ty.RefType               # 引用类型（用于可变变量）
GlobalTypeVar = ty.GlobalTypeVar   # 全局类型变量
TypeCall = ty.TypeCall             # 类型调用（如应用泛型类型）
Any = ty.Any                       # 任意类型（用于类型推断的占位符）

# 导出表达式相关类
Expr = expr.RelayExpr              # 所有表达式的基类
Constant = expr.Constant           # 常量表达式
Tuple = expr.Tuple                 # 元组表达式
Var = expr.Var                     # 变量表达式
GlobalVar = expr.GlobalVar         # 全局变量（用于引用全局函数/类型）
Function = function.Function       # 函数表达式
Call = expr.Call                   # 函数调用表达式
Let = expr.Let                     # 变量绑定表达式（let绑定）
If = expr.If                       # 条件表达式
TupleGetItem = expr.TupleGetItem   # 元组元素访问表达式
RefCreate = expr.RefCreate         # 引用创建表达式（创建可变变量）
RefRead = expr.RefRead             # 引用读取表达式（读取可变变量值）
RefWrite = expr.RefWrite           # 引用写入表达式（修改可变变量值）

# 导出代数数据类型（ADT）相关类
Pattern = adt.Pattern              # 模式匹配的基类
PatternWildcard = adt.PatternWildcard  # 通配符模式（匹配任意值）
PatternVar = adt.PatternVar        # 变量模式（将匹配值绑定到变量）
PatternConstructor = adt.PatternConstructor  # 构造函数模式（匹配ADT构造函数）
PatternTuple = adt.PatternTuple    # 元组模式（匹配元组）
Constructor = adt.Constructor      # ADT构造函数
TypeData = adt.TypeData            # ADT类型定义
Clause = adt.Clause                # 模式匹配子句（模式+表达式）
Match = adt.Match                  # 匹配表达式（类似switch-case）

# 导出辅助函数
var = expr.var                     # 创建变量的便捷函数
const = expr.const                 # 创建常量的便捷函数
bind = expr.bind                   # 绑定变量的便捷函数（简化let表达式）

# 导出类型相关的Functor
TypeFunctor = type_functor.TypeFunctor  # 类型Functor基类
TypeVisitor = type_functor.TypeVisitor  # 类型访问器（用于遍历类型）
TypeMutator = type_functor.TypeMutator  # 类型修改器（用于转换类型）

# 导出表达式相关的Functor
ExprFunctor = expr_functor.ExprFunctor  # 表达式Functor基类
ExprVisitor = expr_functor.ExprVisitor  # 表达式访问器（用于遍历表达式）
ExprMutator = expr_functor.ExprMutator  # 表达式修改器（用于转换表达式）

# 导出Prelude（Relay内置工具函数集）
Prelude = prelude.Prelude

# 再次导出ScopeBuilder（确保作用域构建工具可直接访问）
ScopeBuilder = scope_builder.ScopeBuilder

# 导出参数序列化相关函数
save_param_dict = param_dict.save_param_dict  # 保存模型参数字典
load_param_dict = param_dict.load_param_dict  # 加载模型参数字典
