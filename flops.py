import torch
from thop import profile, clever_format

from mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large


if __name__=="__main__":
    input = torch.randn(1, 3, 224, 224)
    model = MobileNetV3_Small()
    model = MobileNetV3_Large()
    model.eval()

    print(model)

    macs, params = profile(model, inputs=(input, ), custom_ops={})
    macs, params = clever_format([macs, params], "%.3f")

    print('Flops:  ', macs)
    print('Params: ', params)
