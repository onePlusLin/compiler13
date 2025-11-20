from tvm.driver import tvmc

model = tvmc.load('my_model.onnx') # 第 1 步：加载
package = tvmc.compile(model, target="llvm") # 第 2 步：编译
result = tvmc.run(package, device="cpu") # 第 3 步：运行
print(result) 