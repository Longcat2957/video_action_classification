import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, in_channels:int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(1, 1, 1, bias=False)
        self.softmax = nn.Softmax2d()
        self.conv3 = nn.Conv2d(1, in_channels, 1, 1, bias=False)
        
    def forward(self, x:torch.Tensor, attn:torch.Tensor) -> torch.Tensor:
        # attention map
        attn = self.conv2(attn)
        attn = self.softmax(attn)   # for 0 ~ 1 normalization
        
        # feature map
        feat = self.conv1(x)
        feat = F.relu(feat)
        
        # 입력에 attention map을 적용
        out = feat * attn
        
        # restore channels
        out = self.conv3(out)
        out = F.relu(out)
        return out
    
if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 64, 64)
    attn_map = torch.randn(1, 1, 64, 64)
    layer = AttentionModule(3)
    
    b = layer(dummy_input, attn_map)
    print(b.shape)
    