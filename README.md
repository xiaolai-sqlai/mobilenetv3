# A PyTorch implementation of MobileNetV3
I retrain the mobilenetv3 with some novel tricks and [timm](https://github.com/huggingface/pytorch-image-models). 
I also provide the train code, pre-training weight and training logs on this project. 

You should use torch.load to load the model.
```
from mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large

# MobileNetV3_Small
net = MobileNetV3_Small()
net.load_state_dict(torch.load("450_act3_mobilenetv3_small.pth", map_location='cpu'))

# MobileNetV3_Large
net = MobileNetV3_Large()
net.load_state_dict(torch.load("450_act3_mobilenetv3_large.pth", map_location='cpu'))
```

You could reproduce the model by the code.
```
nohup python -u -m torch.distributed.run --nproc_per_node=8 main.py --model mobilenet_v3_small --epochs 300 --batch_size 256 --lr 4e-3 --update_freq 2 --model_ema false --model_ema_eval false --use_amp true --data_path /data/benchmarks/ILSVRC2012 --output_dir ./checkpoint &

nohup python -u -m torch.distributed.run --nproc_per_node=8 main.py --model mobilenet_v3_small --epochs 450 --batch_size 256 --lr 4e-3 --update_freq 2 --model_ema false --model_ema_eval false --use_amp true --data_path /data/benchmarks/ILSVRC2012 --output_dir ./checkpoint &

nohup python -u -m torch.distributed.run --nproc_per_node=8 main.py --model mobilenet_v3_large --epochs 300 --batch_size 256 --lr 4e-3 --update_freq 2 --model_ema false --model_ema_eval false --use_amp true --data_path /data/benchmarks/ILSVRC2012 --output_dir ./checkpoint &

nohup python -u -m torch.distributed.run --nproc_per_node=8 main.py --model mobilenet_v3_large --epochs 450 --batch_size 256 --lr 4e-3 --update_freq 2 --model_ema false --model_ema_eval false --use_amp true --data_path /data/benchmarks/ILSVRC2012 --output_dir ./checkpoint &
```

This is a PyTorch implementation of MobileNetV3 architecture as described in the paper [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf).

Some details may be different from the original paper, welcome to discuss and help me figure it out.

### MobileNetV3
|                       | Madds     | Parameters | Top1-acc  |
| -------------------   | --------- | ---------- | --------- |
| Small (paper)         | 66  M     | 2.9 M      | 67.4%     |
| Small (torchvision)   | 62  M     | 2.5 M      | 67.7%     |
| Small (our 300 epoch) | 69  M     | 3.0 M      | 68.9%     |
| Small (our 450 epoch) | 69  M     | 3.0 M      | 69.2%     |
| Large (paper)         | 219  M    | 5.4 M      | 75.2%     |
| Large (torchvision)   | 235  M    | 5.5 M      | 74.0%     |
| Large (our 300 epoch) | 241  M    | 5.2 M      | 75.6%     |
| Large (our 450 epoch) | 241  M    | 5.2 M      | 75.9%     |
