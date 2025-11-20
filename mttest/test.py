import tvm
from tvm import micro, relay

# 创建 micro target
target = tvm.target.Target("c -keys=cpu -link-params=0")

# 用 micro build
mod = relay.build(...)
