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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The expression functor of Relay."""
from tvm.ir import Op

from .function import Function, FunctionWithFields
from .expr import Call, Let, Var, GlobalVar
from .expr import If, Tuple, TupleGetItem, Constant
from .expr import RefCreate, RefRead, RefWrite
from .adt import Constructor, Match, Clause


class ExprFunctor:
    """
    Expr是？
    An abstract visitor defined over Expr.

    Defines the default dispatch over expressions, and
    implements memoization.
    """

    def __init__(self):
        self.memo_map = {}

    # pylint: disable=no-else-return
    def visit(self, expr):
        '''
        这里做了类型区别，如果是function 就调用visit_function
        如果是call 就调用visit_call
        但是如果一个call的输入是function，那么会导致一些非法访问的bug吧，比如创造LUT op的时候
        '''
        """Apply the visitor to an expression."""
        if expr in self.memo_map:
            return self.memo_map[expr]

        if isinstance(expr, Function):# func应该是到这里吧！先
            # print(":ExprFunctor爷爷:visit, expr is Function,expr:\n",expr)
            res = self.visit_function(expr)
        elif isinstance(expr, Call):# 后来，不知道为什么变成了一个expr
            # print(":ExprFunctor爷爷:visit, expr is Call,expr:\n",expr)# 但是好像是到这？
            res = self.visit_call(expr) # 好像是来创造LUT pass 之后调用来遍历expr从而创造NPU 的LUT op的
        elif isinstance(expr, Let):
            res = self.visit_let(expr)
        elif isinstance(expr, Var):
            res = self.visit_var(expr)
        elif isinstance(expr, GlobalVar):
            res = self.visit_global_var(expr)
        elif isinstance(expr, If):
            res = self.visit_if(expr)
        elif isinstance(expr, Tuple):
            res = self.visit_tuple(expr)
        elif isinstance(expr, TupleGetItem):
            res = self.visit_tuple_getitem(expr)
        elif isinstance(expr, Constant):
            res = self.visit_constant(expr)
        elif isinstance(expr, Op):
            res = self.visit_op(expr)
        elif isinstance(expr, RefCreate):
            res = self.visit_ref_create(expr)
        elif isinstance(expr, RefRead):
            res = self.visit_ref_read(expr)
        elif isinstance(expr, RefWrite):
            res = self.visit_ref_write(expr)
        elif isinstance(expr, Constructor):
            res = self.visit_constructor(expr)
        elif isinstance(expr, Match):
            res = self.visit_match(expr)
        else:
            raise Exception(f"warning unhandled case: {type(expr)}")

        self.memo_map[expr] = res

        return res

    def visit_function(self, _):
        raise NotImplementedError()

    def visit_let(self, _):
        raise NotImplementedError()

    def visit_call(self, _):
        raise NotImplementedError() #当继承的类没实现这个函数是会报错

    def visit_var(self, _):
        raise NotImplementedError()

    def visit_type(self, typ):
        return typ

    def visit_if(self, _):
        raise NotImplementedError()

    def visit_tuple(self, _):
        raise NotImplementedError()

    def visit_tuple_getitem(self, _):
        raise NotImplementedError()

    def visit_global_var(self, _):
        raise NotImplementedError()

    def visit_op(self, _):
        raise NotImplementedError()

    def visit_constant(self, _):
        raise NotImplementedError()

    def visit_ref_create(self, _):
        raise NotImplementedError()

    def visit_ref_write(self, _):
        raise NotImplementedError()

    def visit_ref_read(self, _):
        raise NotImplementedError()

    def visit_constructor(self, _):
        raise NotImplementedError()

    def visit_match(self, _):
        raise NotImplementedError()


class ExprVisitor(ExprFunctor):
    """
    A visitor over Expr.

    The default behavior recursively traverses the AST.
    """

    def visit_tuple(self, tup):
        for x in tup.fields:
            self.visit(x)

    def visit_call(self, call):
        self.visit(call.op)
        for a in call.args:
            self.visit(a)

    def visit_var(self, var):
        pass

    def visit_let(self, let):
        self.visit(let.var)
        self.visit(let.value)
        self.visit(let.body)

    def visit_function(self, fn):
        for x in fn.params:
            self.visit(x)
        self.visit(fn.body)

    def visit_if(self, i):
        self.visit(i.cond)
        self.visit(i.true_branch)
        self.visit(i.false_branch)

    def visit_global_var(self, gv):
        pass

    def visit_constructor(self, c):
        pass

    def visit_op(self, op):
        pass

    def visit_constant(self, const):
        pass

    def visit_ref_create(self, r):
        self.visit(r.value)

    def visit_ref_read(self, r):
        self.visit(r.ref)

    def visit_ref_write(self, r):
        self.visit(r.ref)
        self.visit(r.value)

    def visit_tuple_getitem(self, t):
        self.visit(t.tuple_value)

    def visit_match(self, m):
        self.visit(m.data)
        for c in m.clauses:
            self.visit(c.rhs)

# 爸爸
class ExprMutator(ExprFunctor):
    """
    Expr上的一个功能性访问者。
    默认行为递归遍历AST
    并重构AST。
    A functional visitor over Expr.

    The default behavior recursively traverses the AST
    and reconstructs the AST.
    """
    # 爷爷的visit_function之后到爸爸这里
    def visit_function(self, fn):
        new_params = [self.visit(x) for x in fn.params]
        new_body = self.visit(fn.body)
 
        return FunctionWithFields(fn, list(new_params), new_body)# function是一个不可变对象，需要使用这个函数重新创建一个，所传的参数是需要修改的参数

    def visit_let(self, let):
        new_var = self.visit(let.var)
        new_val = self.visit(let.value)
        new_body = self.visit(let.body)
        return Let(new_var, new_val, new_body)

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        return Call(new_fn, new_args, call.attrs, call.type_args, call.span)

    def visit_var(self, var):
        return var

    def visit_global_id(self, global_var):
        return global_var

    def visit_if(self, ite):
        return If(self.visit(ite.cond), self.visit(ite.true_branch), self.visit(ite.false_branch))

    def visit_tuple(self, tup):
        return Tuple([self.visit(field) for field in tup.fields], tup.span)

    def visit_tuple_getitem(self, op):
        tuple_value = self.visit(op.tuple_value)
        if not tuple_value.same_as(op.tuple_value):
            return TupleGetItem(tuple_value, op.index)
        return op

    def visit_global_var(self, gvar):
        return gvar

    def visit_op(self, op):
        return op

    def visit_constant(self, const):
        return const

    def visit_constructor(self, con):
        return con

    def visit_match(self, m):
        return Match(
            self.visit(m.data),
            [Clause(c.lhs, self.visit(c.rhs)) for c in m.clauses],
            complete=m.complete,
        )

    def visit_ref_create(self, r):
        return RefCreate(self.visit(r.value))

    def visit_ref_write(self, r):
        return RefWrite(self.visit(r.ref), self.visit(r.value))

    def visit_ref_read(self, r):
        return RefRead(self.visit(r.ref))
