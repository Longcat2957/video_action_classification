import os
import torch
import torch.nn as nn
import torchvision
from thop import profile, clever_format

from torchvision.models.video import *

def evaluate_model(model, input_size=(1, 3, 16, 224, 224)):
    '''
        Default Value (B, C, T, H, W)  = (1, 3, 16, 224, 224)
    '''
    input = torch.randn(*input_size)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.2f")
    print('*'*80)
    print(f"# Number of FLOPs: {flops}")
    print(f"# Number of parameters: {params}")
    print('*'*80)


def export_to_onnx(model, input_size=(1, 3, 16, 224, 224), output_path='/home/junghyun/Desktop/wei/mvit_transfer_learning/model2.onnx'):
    model = model.eval()
    dummy_input = torch.randn(*input_size)
    # if not os.path.exists(os.path.dirname(output_path)):
    #     os.makedirs(os.path.dirname(output_path))
    torch.onnx.export(model, dummy_input, output_path, verbose=False, input_names=['input'], output_names=['output'], export_params=True)

def get_model(pretrained:bool=True):
    """
        get torchvision.models.video.mvit_v2
        
        Args:
            pretrained:bool = (default) True
    
        Returns:
            model:nn.Module
    """
    WEIGHTS = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else MViT_V2_S_Weights.DEFAULT
    model = mvit_v2_s(weights=WEIGHTS, progress=True)
    return model

def get_custom_head(num_classes:int=2):
    model_kinetics_400 = get_model(True)
    model_kinetics_400.head = nn.Sequential(
        nn.Dropout1d(), nn.Linear(in_features=768, out_features=num_classes, bias=True)
    )
    return model_kinetics_400

# debug
if __name__ == '__main__':
    mymodel = get_custom_head(2)
    export_to_onnx(mymodel)