import logging
import timm 
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ViT(nn.Module):
  def __init__(self,num_classses, drop_rate = 0.0):
    super(ViT, self).__init__()
    channels = 3
    categories = 4
    self.model = timm.create_model( 'vit_base_patch16_224_miil_in21k',pretrained = True,num_classes = 4, drop_rate = drop_rate)
    num_in_features = self.model.head.in_features
    '''
    self.model.head = nn.Sequential(
              nn.BatchNorm1d(num_features = num_in_features),
              nn.Linear(in_features = num_in_features, out_features = 512,bias = False),
              nn.GELU(),
              nn.BatchNorm1d(num_features = 512),
              nn.Dropout(p=0.4),
              nn.Linear(in_features = 512, out_features = categories, bias = False)
              ) 
    '''

  def forward(self,x):
    return self.model(x)



def build_ViT(num_classes):
    logger.info(f"Model: Vision Transformer-Base 16x16 Imagenet (num_classes) = {num_classes}")
    return ViT(
        num_classses= num_classes,
        drop_rate=0.1
    )

