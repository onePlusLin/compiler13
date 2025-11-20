import sys
import os
import tvm
import numpy as np

# def trace_calls(frame, event, arg):
#     if event == "call":
#         code = frame.f_code
#         print(f"[Python] Call: {code.co_name} at {code.co_filename}:{frame.f_lineno}")
#     return trace_calls

# sys.settrace(trace_calls)
# TVM 源码路径
TVM_SRC_PATH = "/home/linzejia/app/tvm/apache-tvm-src-v0.13.0/"

def trace_calls(frame, event, arg):
    if event == "call":
        code = frame.f_code
        filename = os.path.abspath(code.co_filename)
        # 只跟踪 TVM 源码目录下的调用
        if filename.startswith(TVM_SRC_PATH):
            print(f"[TVM] Call: {code.co_name} at {filename}:{frame.f_lineno}")
    return trace_calls

sys.settrace(trace_calls)

# Demo TVM 操作
import tvm.testing
from tvm import te
import numpy as np

print("start\n")
print("var \n")
n = te.var("n")
print("var done \n")

print("A placeholder \n")
A = te.placeholder((n,), name="A")
print("A placeholder done \n")

print("B placeholder \n")
B = te.placeholder((n,), name="B")
print("B placeholder done\n")

print("C compute \n")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
print("C compute done \n")

print("schedule \n")
s = te.create_schedule(C.op)
print("schedule done \n")

print("target \n")
tgt = tvm.target.Target(target="llvm", host="llvm")
print("target done \n")

print("build \n")
fadd = tvm.build(s, [A, B, C], tgt, name="myadd")
print("build done \n")

print("device \n")
dev = tvm.device(tgt.kind.name, 0)
print("device done \n")

print("ndarray \n")
n = 1024
print("ndarray n done \n")

print("ndarray a \n")
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
print("ndarray a done \n")
print("ndarray b \n")
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
print("ndarray b done \n")


print("ndarray c \n")
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
print("ndarray c done \n")

print("ndarray done \n")
print("fadd \n")
fadd(a, b, c)
print("fadd done \n")

print("assert \n")
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
print("assert done \n")

print("done!")

# import timeit

# np_repeat = 100
# np_running_time = timeit.timeit(
#     setup="import numpy\n"
#     "n = 32768\n"
#     'dtype = "float32"\n'
#     "a = numpy.random.rand(n, 1).astype(dtype)\n"
#     "b = numpy.random.rand(n, 1).astype(dtype)\n",
#     stmt="answer = a + b",
#     number=np_repeat,
# )
# print("Numpy running time: %f" % (np_running_time / np_repeat))

# def evaluate_addition(func, target, optimization, log):
#     dev = tvm.device(target.kind.name, 0)
#     n = 32768
#     a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
#     b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
#     c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

#     evaluator = func.time_evaluator(func.entry_name, dev, number=10)
#     mean_time = evaluator(a, b, c).mean
#     print("%s: %f" % (optimization, mean_time))

#     log.append((optimization, mean_time))

# log = [("numpy", np_running_time / np_repeat)]
# evaluate_addition(fadd, tgt, "naive", log=log)

# print("done!!")
