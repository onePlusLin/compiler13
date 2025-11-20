from tvm.driver import tvmc
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

print("load \n  ")
model = tvmc.load('my_model.onnx') # 第 1 步：加载
print("load done \n ")

print("compile \n ")
package = tvmc.compile(model, target="llvm") # 第 2 步：编译
print("compile done \n ")

print("run \n ")
result = tvmc.run(package, device="cpu") # 第 3 步：运行
print("run done \n ")

print(result) 