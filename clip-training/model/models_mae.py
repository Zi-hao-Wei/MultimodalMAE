# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.layers import DropPath,Mlp

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_image(image, title=''):
    imagenet_mean = np.array([0.4225, 0.4012, 0.3659])
    imagenet_std = np.array([0.2681, 0.2635, 0.2763])
    image = image.detach().cpu().numpy()
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imsave(title, np.clip(np.uint8((image * imagenet_std + imagenet_mean) * 255), 0, 255))
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., scale=1):
        super(Attention,self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 * scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        """
        print("1")
        print(attn[0, 0, 1:196,196:196+13])
        """
        if mask != None:
            # print(attn.shape, mask.shape)
            """
            if attn[0, 0, 1:196,196:196+13].shape[1] < 1:
                print(attn[0, 0])
            """
            attn = attn+mask.unsqueeze(1)
            """
            if attn[0, 0, 1:196,196:196+13].shape[1] < 1:
                print(attn[0, 0])
            """
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        """
        
        print("2")
        # print(attn[0, 0, 1:196,196:196+13])

        if attn[0, 0, 1:196,196:196+13].shape[1] < 1:
            print(attn[0, :, -2, :])
            # plt.imsave("gray.jpg", np.clip(np.uint8((attn[0, 0, 1:13,1:13].transpose(-2,-1)).cpu().detach().numpy()), 0, 1), cmap='gray')
        else:
            print(attn[0, :, 1:196, 199])
            # print(mask[0, 0])

        """


        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block_w_mask(nn.Module):
    def __init__( self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            scale=1):
        super(Block_w_mask, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, scale=scale)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x, mask=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# from timm import to_2tuple
# from

# class PatchEmbed(nn.Module):
#     """ 2D Image to Patch Embedding
#     """
#     def __init__(
#             self,
#             img_size=224,
#             patch_size=16,
#             in_chans=3,
#             embed_dim=768,
#             norm_layer=None,
#             flatten=True,
#             bias=True,
#     ):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x):
#         B, C, H, W = x.shape
#         _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
#         _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x


# def resample_patch_embed(
#         patch_embed,
#         new_size: List[int],
#         interpolation: str = 'bicubic',
#         antialias: bool = True,
#         verbose: bool = False,
# ):
#     """Resample the weights of the patch embedding kernel to target resolution.
#     We resample the patch embedding kernel by approximately inverting the effect
#     of patch resizing.
#     Code based on:
#       https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py
#     With this resizing, we can for example load a B/8 filter into a B/16 model
#     and, on 2x larger input image, the result will match.
#     Args:
#         patch_embed: original parameter to be resized.
#         new_size (tuple(int, int): target shape (height, width)-only.
#         interpolation (str): interpolation for resize
#         antialias (bool): use anti-aliasing filter in resize
#         verbose (bool): log operation
#     Returns:
#         Resized patch embedding kernel.
#     """
#     import numpy as np
#     try:
#         import functorch
#         vmap = functorch.vmap
#     except ImportError:
#         if hasattr(torch, 'vmap'):
#             vmap = torch.vmap
#         else:
#             assert False, "functorch or a version of torch with vmap is required for FlexiViT resizing."

#     assert len(patch_embed.shape) == 4, "Four dimensions expected"
#     assert len(new_size) == 2, "New shape should only be hw"
#     old_size = patch_embed.shape[-2:]
#     if tuple(old_size) == tuple(new_size):
#         return patch_embed

#     if verbose:
#         _logger.info(f"Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation.")

#     def resize(x_np, _new_size):
#         x_tf = torch.Tensor(x_np)[None, None, ...]
#         x_upsampled = F.interpolate(
#             x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
#         return x_upsampled

#     def get_resize_mat(_old_size, _new_size):
#         mat = []
#         for i in range(np.prod(_old_size)):
#             basis_vec = np.zeros(_old_size)
#             basis_vec[np.unravel_index(i, _old_size)] = 1.
#             mat.append(resize(basis_vec, _new_size).reshape(-1))
#         return np.stack(mat).T

#     resize_mat = get_resize_mat(old_size, new_size)
#     resize_mat_pinv = torch.Tensor(np.linalg.pinv(resize_mat.T))

#     def resample_kernel(kernel):
#         resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
#         return resampled_kernel.reshape(new_size)

#     v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
#     return v_resample_kernel(patch_embed)
from model.pos_embed import get_2d_sincos_pos_embed
# from pos_embed import get_2d_sincos_pos_embed
import model.clip as clip


# import clip 
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, vocab_size=49408):
        super().__init__()
        self.vocab_size = vocab_size

        #CLIP Encoder
        self.clip, self.clip_process = clip.load("ViT-B/16")
        self.clip.cuda()
        for param in self.clip.parameters():
            param.requires_grad=False
            
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        #FIXME
        self.patch_embed.num_patches = 196+77
        num_patches = self.patch_embed.num_patches
        # print("num_patches",num_patches)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block_w_mask(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block_w_mask(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, scale=1)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 768, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.bertlinear= nn.Linear(decoder_embed_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.nllloss = nn.NLLLoss(reduce=False)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # print ((self.patch_embed.num_patches),self.pos_embed.data.shape)
        
        #FIXME
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5)+1, cls_token=True)
        pos_embed = pos_embed[:self.patch_embed.num_patches+1,:]

        # print(pos_embed.shape)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5)+1, cls_token=True)
        decoder_pos_embed = pos_embed[:self.patch_embed.num_patches+1,:]

        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, attn_mask=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        if attn_mask != None:
            ids_keep_ = ids_keep+1
            ids_keep_ = torch.cat([torch.zeros((N, 1), device=ids_keep.device, dtype=ids_keep.dtype), ids_keep_], dim=1)
            print(ids_keep_)
            D_ = attn_mask.shape[1]
            attn_mask = torch.gather(attn_mask, dim=1, index=ids_keep_.unsqueeze(-1).repeat(1, 1, D_))
            D_ = attn_mask.shape[1]
            attn_mask = torch.gather(attn_mask, dim=2, index=ids_keep_.unsqueeze(1).repeat(1, D_, 1))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, attn_mask

    def forward_encoder(self, image_features,text_features, mask_ratio_img, mask_ratio_text, attn_mask=None):
        # embed patches
        # x = self.patch_embed(x)

        # add pos embed w/o cls token
        # FIXME
        image_features = image_features + self.pos_embed[:, 1:197, :]
        text_features = text_features + self.pos_embed[:, 197:, :]
        
        image_features_ = image_features
        
        if attn_mask != None:
            pass
            attn_mask = attn_mask[:,196:,196:]
        
        # masking: length -> length * mask_ratio
        image_features, mask1, ids_restore1, _ = self.random_masking(image_features, mask_ratio_img)
        text_features, mask2, ids_restore2, attn_mask = self.random_masking(text_features, mask_ratio_text, attn_mask)

        # print(ids_restore1)
        # print(mask1)
        x = torch.cat([image_features,text_features],1)

        mask = torch.cat([mask1,mask2],1)
        mask=mask1
        mask2 = None
        #FIXME
        ids_restore2 = ids_restore2 + 196
        ids_restore = torch.cat([ids_restore1,ids_restore2],1)
        # ids_restore = ids_restore1
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 

        # print(x.shape)
        # apply Transformer blocks
        # print(x.shape,attn_mask.shape)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.norm(x)
        
        
        
        # sim = torch.bmm(x[:,1:14]/10,image_features_[:,:].transpose(-2,-1)).softmax(-1)
        # print(sim[0])
        
        # return x, mask, mask1, mask2, ids_restore, sim
        
        
        
        return x, mask, mask1, mask2, ids_restore

    def forward_decoder(self, x, ids_restore, attn_mask=None):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, attn_mask)
        x = self.decoder_norm(x)
        # remove cls token
        x = x[:, 1:, :]

        # predictor projection

        img_pred = self.decoder_pred(x[:,:196,:])
        text_pred = self.softmax(self.bertlinear(x[:,196:,:]))


        return img_pred,text_pred

    def forward(self, imgs, text, img_mask_ratio=1, text_mask_ratio=0.25, attn_mask = None):
        attn_mask = attn_mask.to(imgs.device)
        image_features = self.clip.encode_image(imgs)
        text_features = self.clip.encode_text(text)

        # normalized features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)



        unified_features = torch.cat([image_features,text_features],1)

        latent, unified_mask, img_mask, token_mask, ids_restore = self.forward_encoder(image_features, text_features, img_mask_ratio, text_mask_ratio, attn_mask)
        """
        for i in range(imgs.shape[-2]):
            for j in range(imgs.shape[-1]):
                idx_ = (i // 16) * 14 + j // 16
                imgs[0, 0, i, j] += 10 * sim[0, 2, idx_]
                imgs[0, 1, i, j] = 0
                imgs[0, 2, i, j] = 0
        show_image(imgs.squeeze(0).permute(1,2,0), 'highlighted.jpg')
        exit()
        """
        img_pred, text_pred = self.forward_decoder(latent, ids_restore, attn_mask)  # [N, L, p*p*3]
        img_loss = self.forward_img_loss(imgs, img_pred, img_mask)
        
        # show_image(imgs.squeeze(0).permute(1,2,0), 'orignal.jpg')
        # show_image(self.unpatchify(img_pred).squeeze(0).permute(1,2,0), 'reconstructed.jpg')
        # text_loss = self.forward_text_loss(text, text_pred, token_mask)
        # loss = img_loss + text_loss
        loss = img_loss
        # exit()
        # loss = ((img_pred - image_features)**2).mean(-1).mean()
        # print(loss)

        return loss


    # def forward(self, imgs, mask_ratio=0.75):
    #     latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
    #     pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
    #     loss = self.forward_loss(imgs, pred, mask)
    #     return loss, pred, mask

    def forward_img_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        
        """
        U, S, V = torch.linalg.svd(target, full_matrices=True)
        U_, S_, V_ = torch.linalg.svd(pred, full_matrices=True)
        print(U.shape, S.shape, V.shape)
        first = torch.outer(U[0, :, 0], V[0, 0, :])*S[0,0]
        first_pred = torch.outer(U_[0, :, 0], V_[0, 0, :])*S_[0,0]
        
        print(first.shape)
        show_image(self.unpatchify(first.unsqueeze(0)).squeeze(0).permute(1,2,0), 'first_sing.jpg')
        show_image(self.unpatchify(first_pred.unsqueeze(0)).squeeze(0).permute(1,2,0), 'first_sing_pred.jpg')
        result = []
        for i in range(12):
            diff = (U[:, :, i] - U_[:, :, i]) ** 2
            result.append((diff.detach().cpu().numpy().sum()/ 512).round(4))
        print(result)
        """

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        # print("pred",pred.shape)
        # print("target",target.shape)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_text_loss(self, tokens, pred, mask):
        """
        """
        #Need Mapping pred to labels
        # print("tokens",tokens)

        # tokens = pred.reshape(-1,77)
        # pred = pred.reshape(-1,self.vocab_size)
        pred = pred.transpose(1,2)
        # print("pred",pred.shape)
        tokens = tokens.long()
        loss = self.nllloss(pred, tokens)
        # print("Loss", loss.shape)

        loss = torch.sum(loss * mask) / (torch.sum(mask) + 1e-10)
        return loss

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=512, depth=12, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
