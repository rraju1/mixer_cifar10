# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torchsummary
import numpy as np
import json
import warnings
from torch import Tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

class MaskedViT(VisionTransformer):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0, 
                 drop_path_rate=0, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, 
                        qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, **kwargs)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.device = kwargs['device']
        self.lambda_drop = kwargs['lambda_drop']
        self.attn_maps_path = kwargs['attn_maps_path']
        self.attn_maps_test_path = kwargs['attn_maps_test_path']
        self.phase = kwargs['phase']
        self.patchdroptest = kwargs['patchdroptest']
        
        with open(self.attn_maps_path) as json_file:
            self.data = json.load(json_file)
        with open(self.attn_maps_test_path) as json_file:
            self.test_data = json.load(json_file)

        self.unfold_fn = nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)
        self.A = torch.randn(embed_dim, in_chans * patch_size * patch_size).to(device=self.device)
    
    def forward(self, x, idx=None):
        x_tokens = self.prepare_tokens(x)
        if self.phase == 'train':
            x = self.patchdrop(x_tokens, x, idx, self.data, self.lambda_drop)
        elif self.phase == 'test' and self.patchdroptest:
            x = self.patchdrop(x_tokens, x, idx, self.test_data, self.lambda_drop)
        else:
            x = x_tokens

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]
    
    # gets called after prepare tokens and positional info is added
    def patchdrop(self, x, img, idx, attn_dict, drop_lambda):
        B, N, C = x.shape
        new_N = int((1 - drop_lambda) * (N - 1)) # need to discard cls token in calculation
        end_params = B * new_N * C
        x_ncls = x[:,1:,:] + 1e-8 # extract all tokens exlcuding cls token, need to add 1e-8 to make sure pos emb is not zero
        mask = self.project_bin_mask(img, idx, attn_dict, drop_lambda, new_N)
        masked_x = x_ncls * mask
        
        masked_x = masked_x[masked_x != 0]
        
        # need to deal with the case where drop didnt work

        if masked_x.shape[0] == end_params:
            new_input = masked_x.reshape(B, new_N, C)
        elif masked_x.shape[0] < end_params: # too many patches were dropped
            padded_value = torch.zeros(end_params - masked_x.shape[0]).unsqueeze(0).to(self.device)
            new_input = torch.cat([masked_x, padded_value]).reshape(B, new_N, C)
        else: # too many patches were retained
            new_input = masked_x[:end_params].reshape(B, new_N, C)

        # print(new_input.shape)
        assert new_input.shape == (B, new_N, C)
        new_input = torch.cat([x[:,0,:].unsqueeze(1), new_input], dim=1)
        # print(new_input.shape)
        return new_input
        
    def project_bin_mask(self, image, index, data, lambda_drop, new_N):
        mask = self.get_mask_batch(image, index, data, lambda_drop, new_N)
        patched = self.gen_mask(mask, self.unfold_fn)
        output = torch.nn.functional.linear(patched, self.A, bias=None)
        bin_output = self.create_binary_mask(output)
        return bin_output
    
    """
    Given tensor x, create a binary masks removing the effect of multiplying by random linear transform A
    """
    def create_binary_mask(self, x):
        zeros = torch.zeros_like(x)
        return zeros.eq(x).bitwise_not_().float()

    def gen_mask(self, masks, unfold_fn):
        patched_tensor = unfold_fn(masks.repeat(1,3,1,1))
        patched_tensor = patched_tensor.permute(0,2,1)
        return patched_tensor

    def get_mask_batch(self, image, idx, attn_dict, drop_lambda, new_N):
        idx_np = idx.numpy()
        w_featmap = int(np.sqrt(len(attn_dict[str(0)]))) # 14 0 is a random key
        h_featmap = int(np.sqrt(len(attn_dict[str(0)]))) # 14
        scale = image.shape[2] // w_featmap # to pass to interpolate
        batch_size = len(idx)

        batch_array = [] # collect attn maps
        for i in range(batch_size):
            batch_array.append(np.array(attn_dict[str(idx_np[i])]))
        batch_tensor = torch.from_numpy(np.vstack(batch_array)).to(self.device)

        val, indices = torch.sort(batch_tensor, dim=1)
        # assuming nonsalient patchdrop/random
        threshold = torch.quantile(val, drop_lambda, dim=1)
        th_attn = val >= threshold[:,None]
        # salient patchdrop
        # threshold = torch.quantile(val, (1 - drop_lambda), dim=1)
        # th_attn = val <= threshold[:,None]
        total_N = th_attn.shape[1]
        ascend_N = total_N - new_N
        th_attn[:,:ascend_N] = False
        th_attn[:,ascend_N:] = True
        
        idx2 = torch.argsort(indices, dim=1) # rearrange patch positions
        for batch_idx in range(th_attn.shape[0]):
            th_attn[batch_idx] = th_attn[batch_idx][idx2[batch_idx]]

        th_attn = th_attn.float() # bool -> float
        bin_mask = th_attn.reshape(-1, w_featmap, h_featmap)
        mask = torch.nn.functional.interpolate(bin_mask.unsqueeze(1), scale_factor=scale, mode="nearest")
        return mask

    def get_last_selfattention(self, x, idx=None):
        x_tokens = self.prepare_tokens(x)
        if self.phase == 'train':
            x = self.patchdrop(x_tokens, x, idx, self.data, self.lambda_drop)
        elif self.phase == 'test' and self.patchdroptest:
            x = self.patchdrop(x_tokens, x, idx, self.test_data, self.lambda_drop)
        else:
            x = x_tokens

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1, idx=None):
        x_tokens = self.prepare_tokens(x)
        if self.phase == 'train':
            x = self.patchdrop(x_tokens, x, idx, self.data, self.lambda_drop)
        elif self.phase == 'test' and self.patchdroptest:
            x = self.patchdrop(x_tokens, x, idx, self.test_data, self.lambda_drop)
        else:
            x = x_tokens

        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def set_patchdrop_test(self, use_idx_test):
        self.patchdroptest = use_idx_test

    def set_phase(self, new_phase):
        self.phase = new_phase
        
    def set_lambda(self, lambda_drop):
        self.lambda_drop = lambda_drop
    
    def get_lambda(self):
        return self.lambda_drop
    
    def get_phase(self):
        return self.phase
    
    def get_patchdrop_test(self):
        return self.patchdroptest

class DynVit(VisionTransformer):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0, 
                 drop_path_rate=0, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, 
                        qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, **kwargs)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.device = kwargs['device']
        self.lambda_drop = kwargs['lambda_drop']
        self.phase = kwargs['phase']
        self.patchdroptest = kwargs['patchdroptest']

        self.num_patches = int((img_size[0] * img_size[0])/(patch_size * patch_size))
        # self.pooling_op = nn.AvgPool1d(self.embed_dim)

        self.pooling_op = nn.AvgPool1d(self.embed_dim)
        self.pdp_mod = nn.Sequential(
                nn.Linear(self.num_patches, self.num_patches)
        )
    
    def forward(self, x):
        x_tokens = self.prepare_tokens(x)
        if self.phase == 'train':
            x = self.patchdrop(x_tokens)
        elif self.phase == 'test' and self.patchdroptest:
            x = self.patchdrop(x_tokens)
        else:
            x = x_tokens
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]
    
    # gets called after prepare tokens and positional info is added
    def patchdrop(self, x):
        B, N, C = x.shape
        lambda_drop = self.get_lambda()
        new_N = int((N-1) * (1 - lambda_drop))
        end_params = B*new_N*C

        x_ncls = x[:,1:,:] + 1e-8 # extract all tokens exlcuding cls token, need to add 1e-8 to make sure pos emb is not zero
        bin_mask = self.patchdrop_module(x_ncls, new_N)

        masked_x = x_ncls * bin_mask
        masked_x = masked_x[masked_x != 0]

        # need to deal with the case where drop didnt work

        if masked_x.shape[0] == end_params:
            new_input = masked_x.reshape(B, new_N, C)
        elif masked_x.shape[0] < end_params: # too many patches were dropped
            padded_value = torch.zeros(end_params - masked_x.shape[0]).unsqueeze(0).to(self.device)
            new_input = torch.cat([masked_x, padded_value]).reshape(B, new_N, C)
        else: # too many patches were retained
            new_input = masked_x[:end_params].reshape(B, new_N, C)

        # print(new_input.shape)
        assert new_input.shape == (B, new_N, C)

        new_input = torch.cat([x[:,0,:].unsqueeze(1), new_input], dim=1)
        return new_input

    def patchdrop_module(self, x, new_N):
        lambda_drop = self.get_lambda()
        x_pool = self.pooling_op(x).squeeze(-1)
        saliency = self.pdp_mod(x_pool)
        val, indices = torch.sort(saliency, dim=1)
        threshold = torch.quantile(val.float(), lambda_drop, dim=1)
        th_attn = val >= threshold[:,None]
        idx2 = torch.argsort(indices, dim=1)

        total_N = th_attn.shape[1]
        ascend_N = total_N - new_N
        th_attn[:,:ascend_N] = False
        th_attn[:,ascend_N:] = True

        for batch_idx in range(th_attn.shape[0]):
            th_attn[batch_idx] = th_attn[batch_idx][idx2[batch_idx]]
        th_attn = th_attn.unsqueeze(-1)
        new_input = th_attn.repeat(1,1,self.embed_dim)
        return new_input.float()

    def set_patchdrop_test(self, use_idx_test):
        self.patchdroptest = use_idx_test

    def set_phase(self, new_phase):
        self.phase = new_phase
        
    def set_lambda(self, lambda_drop):
        self.lambda_drop = lambda_drop
    
    def get_lambda(self):
        return self.lambda_drop
    
    def get_phase(self):
        return self.phase
    
    def get_patchdrop_test(self):
        return self.patchdroptest

def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(**kwargs):
    model = VisionTransformer(**kwargs)
    return model


def vit_small_patchdrop(**kwargs):
    model = MaskedViT(**kwargs)
    return model

def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_dynamic(**kwargs):
    model = DynVit(**kwargs)
    return model

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x