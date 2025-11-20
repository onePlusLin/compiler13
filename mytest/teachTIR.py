import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

import numpy as np

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (8,), dtype="float32")
        B = T.match_buffer(b, (8,), dtype="float32")
        for i in range(8):
            # A block is an abstraction for computation.
            with T.block("B"):
                # Define a spatial block iterator and bind it to value i.
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0


ir_module = MyModule
print(type(ir_module))

# <class 'tvm.ir.module.IRModule'>

mod = tvm.build(ir_module, target="c")
# mod = tvm.build(ir_module, target="llvm")
# mod = tvm.build(ir_module, target="cuda")

a = tvm.nd.array(np.arange(8).astype("float32"))
print(a)
# [0. 1. 2. 3. 4. 5. 6. 7.]

b = tvm.nd.array(np.zeros((8,)).astype("float32"))
mod(a, b)
print(b)
# [1. 2. 3. 4. 5. 6. 7. 8.]