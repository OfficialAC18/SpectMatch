import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from timm.models.layers import to_2tuple, trunc_normal_
import logging

logger = logging.getLogger(__name__)


#To override timm input shape constraint
class PatchEmbed(nn.Module):
  def __init__(self, img_size = 224, patch_size = 16, in_chans = 3, embed_dim = 768):
    super().__init__()


    img_size = to_2tuple(img_size)
    patch_size = to_2tuple(patch_size)
    num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
    self.img_size = img_size
    self.patch_size = patch_size
    self.num_patches = num_patches


    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size)

  def forward(self, x):
    x = self.proj(x).flatten(2).transpose(1, 2)
    return x


class ASTModel(nn.Module):
  """
  AST
  label_dim -> Number of Total Classes (For Classroom, = 4)
  fstride -> patch_splitting_stride for freq, for 16*16, fstride = 16 (No Overlap), fstride = 10 (6 Overlap)
  tstride -> patch_splitting_stride for time, for 16*16, fstride = 16 (No Overlap), fstride = 10 (6 Overlap)
  input_fdim -> Num Of Frequency Bins Of Spectrogram (For Classroom = 128)
  input_tdim -> Num Of Time Frames Of Spectrogram (For Classroom = 298)
  imagenet_pretrain -> Use Imagenet Pretrained Model (For Classroom = True)
  audioset_pretrain -> Use Both Imagenet and Audioset (For Classroom = False, (Following SpeechCommands Pattern))
  model_size  -> AST Model Sizes, [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
  """

  def __init__(self, label_dim = 4, fstride = 10, tstride = 10, input_fdim = 128, input_tdim = 298, imagenet_pretrain = True, audioset_pretrain = False, model_size = 'base384'):

    super(ASTModel, self).__init__()
    assert timm.__version__ == '0.4.5' #Might Not Be Compatible With Other Versions, Hence Use This

    timm.models.vision_transformer.PatchEmbed = PatchEmbed #Using Our Custom Defined PatchEmbed

    def get_shape(self, fstride, tstride, input_fdim = input_fdim, input_tdim = input_tdim):
      test_input = torch.randn(1, 1, input_fdim, input_tdim)
      test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size = (16,16), stride = (fstride, tstride))
      test_out = test_proj(test_input)
      f_dim = test_out.shape[2]
      t_dim = test_out.shape[3]
      return f_dim, t_dim



    if audioset_pretrain == False:
      if model_size == 'tiny224':
        self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
      elif model_size == 'small224':
        self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
      elif model_size == 'base224':
        self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
      elif model_size == 'base384':
        self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
      else:
        raise Exception('Model size must be one of tiny224, small224, base224, base384.')

      self.original_num_patches = self.v.patch_embed.num_patches
      self.original_hw = int(self.original_num_patches ** 0.5)
      self.original_embedding_dim = self.v.pos_embed.shape[2]
      self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))




      #Automatically Get Intermediate Shape
      #Point of Change
      f_dim, t_dim =  get_shape(self, fstride, tstride)
      num_patches =  f_dim * t_dim
      self.v.patch_embed.num_patches = num_patches


      #Linear Projection Layer
      new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size = (16,16), stride = (fstride, tstride))
      if imagenet_pretrain == True:
        #We Are Doing it This Way Because ViT uses 3 Channel Images, whereas we are using 1 Channel Audio
        new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim = 1).unsqueeze(1))
        new_proj.bias = self.v.patch_embed.proj.bias
      self.v.patch_embed.proj = new_proj

      #To Obtain Positional Embedding
      if imagenet_pretrain == True:
        # get positional embedding from deit model, skip first 2 tokens (cls and distillation), reshape to orignal 2D Shape
        new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1,2).reshape(1, self.original_embedding_dim, self.original_hw, self.original_hw)
        # cut (from middle) or interpolate the second dimension of the positional embedding
        if t_dim <= self.original_hw:
          new_pos_embed = new_pos_embed[:, :, :,int(self.original_hw / 2) - int(t_dim/2): int(self.original_hw / 2) - int(t_dim / 2) + t_dim]
        else:
          new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.original_hw, t_dim), mode = 'bilinear')
        # cut (from middle) or interpolate the first dimension of the positional embedding
        if f_dim <= self.original_hw:
          new_pos_embed = new_pos_embed[:, :, int(self.original_hw / 2) - int(f_dim / 2): int(self.original_hw / 2) - int(f_dim / 2) + f_dim, :]
        else:
          new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size = (self.f_dim, self.t_dim), mode = 'bilinear')
        #Flatten Positional Embedding
        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
        #Concatenate The Above Positional Embedding With The CLS Token and Distillation Token of The DeiT Model.
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
      else:
        new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim)) 
        self.v.pos_embed = new_pos_embed
        trunc_normal_(self.v.pos_embed, std=.02)

    elif audioset_pretrain == True:
      if audioset_pretrain == True and imagenet_pretrain == False:
        raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
    
      if model_size != 'base384':
        raise ValueError('currently only has Audioset base384 pretrained model.')
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      if os.path.exists('../pretrained_models/audioset_10_10_0.4593.pth') == False:
        audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
        wget.download(audioset_mdl_url, out='../pretrained_models/audioset_10_10_0.4593.pth')
      sd = torch.load('../pretrained_models/audioset_10_10_0.4593.pth', map_location=device)
      audio_model = ASTModel(label_dim=4, fstride=10, tstride=10, input_fdim=128, input_tdim=298, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
      audio_model = torch.nn.DataParallel(audio_model)
      audio_model.load_state_dict(sd, strict=False)
      self.v = audio_model.module.v
      self.original_embedding_dim = self.v.pos_embed.shape[2]
      self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

      f_dim, t_dim = get_shape(self, fstride, tstride)
      num_patches = f_dim*t_dim
      self.v.patch_embed.num_patches = num_patches
      new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1,2).reshape(1, 768, 12, 101)
      # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
      if t_dim < 101:
        new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
      # otherwise interpolate
      else:
        new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
      new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
      self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

      
  @autocast()
  def forward(self, x):
    """
    :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
    :return: prediction
    """

    x = x.unsqueeze(1)
    x = x.transpose(2, 3)

    B = x.shape[0]
    x = self.v.patch_embed(x)
    cls_tokens = self.v.cls_token.expand(B, -1, -1)
    dist_token = self.v.dist_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, dist_token, x), dim = 1)
    x = x + self.v.pos_embed
    x = self.v.pos_drop(x)
    for blk in self.v.blocks:
      x = blk(x)
    x = self.v.norm(x)
    x = (x[:, 0] + x[:, 1]) / 2

    x = self.mlp_head(x)
    return x

def build_AST(num_classes):
    logger.info(f"Model: Audio Spectrogram Transformer (num_classes = {num_classes})")
    model =  ASTModel(label_dim=num_classes)
    return model











