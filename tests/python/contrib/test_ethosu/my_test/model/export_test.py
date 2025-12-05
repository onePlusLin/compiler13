import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import os

print("TensorFlow version:", tf.__version__)
# 输出文件名
tflite_file_name = "toy_multi_branch_int8.tflite"

# 构建我的模型
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # === 关键修改：定义 4 组独立的层，以构建 4 条独立的计算路径 ===
        # 如果复用同一个 self.conv2d，生成的 TFLite 模型只会有一条路径，没法做对比测试
        
        # 分支 0: 正常的 SiLU (Conv -> Sigmoid -> Mul)
        self.conv0 = layers.Conv2D(32, (6, 6), (2, 2), 'valid', use_bias=True)
        self.sig0  = layers.Activation('sigmoid')
        self.mul0  = layers.Multiply()
        
        # 分支 1: 单纯的 Sigmoid (Conv -> Sigmoid) -> 用于测试单层逻辑
        self.conv1 = layers.Conv2D(32, (6, 6), (2, 2), 'valid', use_bias=True)
        self.sig1  = layers.Activation('sigmoid')
        
        # 分支 2: 共享输入乘法 (Conv -> Mul(x, x)) -> 测试 z*z 结构
        self.conv2 = layers.Conv2D(32, (6, 6), (2, 2), 'valid', use_bias=True)
        self.mul2  = layers.Multiply()
        
        # 分支 3: 另一个正常的 SiLU (用于做对比，比如这条路开双流，分支0不开)
        self.conv3 = layers.Conv2D(32, (6, 6), (2, 2), 'valid', use_bias=True)
        self.sig3  = layers.Activation('sigmoid')
        self.mul3  = layers.Multiply()
    
    def call(self, inputs):
        # 路径 0
        x0 = self.conv0(inputs)
        s0 = self.sig0(x0)
        result0 = self.mul0([x0, s0]) 
        
        # 路径 1 (单分支测试)
        x1 = self.conv1(inputs)
        result1 = self.sig1(x1)
        
        # 路径 2 (共享参数乘法测试 x*x)
        x2 = self.conv2(inputs)
        result2 = self.mul2([x2, x2]) 
        
        # 路径 3 (正常乘法备份)
        x3 = self.conv3(inputs)
        s3 = self.sig3(x3)
        result3 = self.mul3([x3, s3])
        
        # 返回列表，TFLite 会生成 4 个输出 Output Tensor
        return [result0, result1, result2, result3]

# --- 模拟输入 ---
input_shape = (1, 640, 640, 3)
dummy_input = tf.random.normal(input_shape)

model = MyModel()
outputs = model(dummy_input) # 注意：返回的是一个列表 list

print("\n=== 模型构建成功 ===")
print(f"输入形状: {input_shape}")
# 修正：outputs 是一个 list，不能直接调 .shape，要遍历打印
for i, out in enumerate(outputs):
    print(f"输出 {i} 形状: {out.shape}")

model.summary()

# 3. 准备量化所需的 Representative Dataset
def representative_data_gen():
    for _ in range(10): # 跑 10 次够了，太慢
        # 【修正】尺寸必须和模型输入 (644, 644) 一致！
        # 之前的 (16, 16) 会导致量化统计信息严重错误
        data = np.random.randint(0, 255, (1, 640, 640, 3)).astype(np.float32) / 255.0
        yield [data]

# --- 主流程 ---

# 2. 转换为 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. 开启 Int8 全整型量化
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# 4. 强制所有算子使用 Int8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# 5. 设置输入输出类型
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.int8

print("开始转换模型 (这可能需要一点时间)...")
tflite_model = converter.convert()

# 6. 保存文件
with open(tflite_file_name, "wb") as f:
    f.write(tflite_model)

print(f"\n✅ Success! Generated {tflite_file_name}")
print(f"File size: {os.path.getsize(tflite_file_name) / 1024:.2f} KB")

# 7. 验证
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

print("\n--- Model Details ---")
input_details = interpreter.get_input_details()[0]
print(f"Input: {input_details['name']}, Type: {input_details['dtype']}")

# 打印所有输出的信息
output_details = interpreter.get_output_details()
for i, detail in enumerate(output_details):
    print(f"Output {i}: {detail['name']}, Type: {detail['dtype']}, Scale: {detail['quantization'][0]}, ZP: {detail['quantization'][1]}")