import torch
from torch import nn, einsum
from einops import rearrange
import logging

logger = logging.getLogger(__name__)


PATH = "/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/Best Model - Audio MAE /encoder.pt"

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(1024, 128), patch_size=(16, 16), in_chans=1, embed_dim=768):
        super().__init__()

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
      super().__init__()
      self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
      )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViTAudioMAE(nn.Module):

  def __init__(self, num_mels=128, mel_len=1024, patch_size=16, in_chans=1,
                 embed_dim=768, encoder_depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4, norm_layer=nn.LayerNorm, norm_pix_loss=False):

    super().__init__()
    #Initializing Everything First
    self.patch_embed = PatchEmbed((mel_len, num_mels), (patch_size, patch_size), in_chans, embed_dim)
    num_patches = self.patch_embed.num_patches
    self.pos_embed = nn.Parameter(torch.zeros(1,num_patches + 1, embed_dim),requires_grad = False)
    self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))

    self.encoder = Transformer(embed_dim, encoder_depth, num_heads, embed_dim // num_heads, mlp_ratio * embed_dim)

    self.norm = norm_layer(embed_dim)
    #Load Checkpoint
    checkpoint = torch.load(PATH)

    #Loading the weights of the different parts of the encoder
    self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    self.patch_embed.load_state_dict(checkpoint['patch_embed_state_dict'])
    self.pos_embed = checkpoint['pos_embed_state_dict']
    self.cls_token = checkpoint['cls_token_state_dict']
    self.norm.load_state_dict(checkpoint['norm_layer_state_dict'])

    #Freezing the layers of the encoder
    '''
    for layer in self.encoder.parameters():
      layer.requires_grad = False
    '''

    #Creating classification head
    self.flatten = nn.Flatten()
    #self.head = nn.Linear(embed_dim*103, 4)
    self.head = nn.Linear(393984, 4,)

  def forward(self,x):
    #Perform Patch Embedding
    x = self.patch_embed(x)

    #Add Positional Embedding
    x = x + self.pos_embed[:, 1:, :]

    #Append the CLS token
    cls_token = self.cls_token + self.pos_embed[:,:1,:]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim = 1)
    torch.cuda.empty_cache()


    #Pass through Transformer Layers
    x = self.encoder(x)
    x = self.norm(x)

    #Flatten and pass through fully connected layer
    x = self.flatten(x)
    x = self.head(x)

    return x


def build_ViTMAE(num_classes):
    logger.info(f"Model: ViT-Audio-MAE (num_classes = {num_classes})")
    model = ViTAudioMAE()
    return model

















