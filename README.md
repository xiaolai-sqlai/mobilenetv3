# A PyTorch implementation of MobileNetV3
I make a mistake to forget the avgpool in se model, now I have re-trained the mbv3_small, the mbv3_large is on training, it will be coming soon.

You should use torch.load to load the model.

···
model = torch.load("mbv3_large.old.pth.tar", map_location='cpu')
weight = model["state_dict"]
···

This is a PyTorch implementation of MobileNetV3 architecture as described in the paper [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf).

Some details may be different from the original paper, welcome to discuss and help me figure it out.

### MobileNetV3
|              | Madds     | Parameters | Top1-acc  |
| -----------  | --------- | ---------- | --------- |
| Large        | 219 M     | 5.4  M     | 75.2%     |
| Small        | 66  M     | 2.9  M     | 67.4%     |
| Ours Large old   | 272 M     | 3.96  M     | 75.454%   |
| Ours Small old   | 66  M     | 2.51  M     | 69.069%   |
| Ours Large new   | 265 M     | 3.96  M     |   |
| Ours Small new   | 64  M     | 2.51  M     | 69.037%   |
