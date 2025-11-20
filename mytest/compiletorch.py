import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata

# 导入 PyTorch
import torch
import torchvision

model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# 通过追踪获取 TorchScripted 模型
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()