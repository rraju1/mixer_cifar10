import numpy as np
import torch
import torch.nn as nn
import warnings
import math

def get_model(args):
    model = None
    if args.model=='mlp_mixer':
        from utils.mlp_mixer import MLPMixer
        model = MLPMixer(
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size = args.patch_size,
            hidden_c = args.hidden_c,
            hidden_s = args.hidden_s,
            num_layers = args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act,
            is_cls_token=args.is_cls_token
        )
    elif args.model=='mlp_mixer_masked':
        from utils.mlp_mixer import MaskedMLPMixer
        model = MaskedMLPMixer(
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size = args.patch_size,
            hidden_c = args.hidden_c,
            hidden_s = args.hidden_s,
            num_layers = args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act,
            is_cls_token=args.is_cls_token
        )
    elif args.model=='vit_small':
        from utils.vision_transformer import vit_small
        kwargs = {}
        kwargs['in_chans'] = 3
        kwargs['embed_dim'] = args.hidden_size
        kwargs['depth'] = args.num_layers
        kwargs['num_heads'] = args.num_heads
        kwargs['mlp_ratio'] = args.mlp_ratio
        kwargs['qkv_bias'] = args.qkv_bias
        kwargs['img_size'] = [args.size]
        kwargs['num_classes'] = args.num_classes
        kwargs['qk_scale'] = args.qk_scale
        kwargs['drop_rate'] = args.drop_p
        kwargs['attn_drop_rate'] = args.attn_drop_rate
        kwargs['drop_path_rate'] = args.drop_path_rate
        kwargs['patch_size'] = args.patch_size
        model = vit_small(**kwargs)
    else:
        raise ValueError(f"No such model: {args.model}")

    return model.to(args.device)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def project_bin_mask(image, index, data, lambda_drop, ps, in_chan, hidden):
    A = torch.randn(hidden, in_chan * ps * ps)
    unfold_fn = nn.Unfold(kernel_size=(ps, ps), stride=ps)
    mask = get_mask_batch(image, index, data, lambda_drop)
    patched = gen_mask(mask, unfold_fn)
    output = torch.nn.functional.linear(patched, A, bias=None)
    return output

def get_mask_batch(image, idx, attn_dict, drop_lambda):
    idx_np = idx.numpy()
    w_featmap = int(np.sqrt(len(attn_dict[str(0)]))) # 14 0 is a random key
    h_featmap = int(np.sqrt(len(attn_dict[str(0)]))) # 14
    scale = image.shape[2] // w_featmap # to pass to interpolate
    batch_size = len(idx)

    batch_array = [] # collect attn maps
    for i in range(batch_size):
        batch_array.append(np.array(attn_dict[str(idx_np[i])]))
    batch_tensor = torch.tensor(np.array(batch_array))

    val, indices = torch.sort(batch_tensor, dim=1)
    threshold = torch.quantile(val, drop_lambda, dim=1)
    th_attn = val >= threshold[:,None]
    idx2 = torch.argsort(indices, dim=1) # rearrange patch positions
    for batch_idx in range(th_attn.shape[0]):
        th_attn[batch_idx] = th_attn[batch_idx][idx2[batch_idx]]

    th_attn = th_attn.float() # bool -> float
    bin_mask = th_attn.reshape(-1, w_featmap, h_featmap)
    mask = torch.nn.functional.interpolate(bin_mask.unsqueeze(1), scale_factor=scale, mode="nearest")
    return mask

def gen_mask(masks, unfold_fn):
    patched_tensor = unfold_fn(masks.repeat(1,3,1,1))
    patched_tensor = patched_tensor.permute(0,2,1)
    return patched_tensor