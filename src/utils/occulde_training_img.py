import torch
import matplotlib.pyplot as plt
import numpy as np

from types import SimpleNamespace
from collections import OrderedDict
from dataloader import get_dataloaders
from utils.vision_transformer import vit_base, DINOHead
from torch.utils.data import Dataset
from PIL import Image
import json


args = SimpleNamespace()
args.dataset = 'c10'
args.model = 'mlp_mixer'
args.batch_size = 32
args.eval_batch_size = 10
args.num_workers = 4
args.seed = 0
args.epochs = 300
args.patch_size = 4
args.autoaugment = False
args.use_cuda = False
args.size = 224
args.split = 'index'

train_dataloader, test_dataloader = get_dataloaders(args)

model = vit_base(num_classes=100)
state_dict = torch.load('cifar100_ViT_B_dino.pth')

stripped_keys = OrderedDict()
for k, v in state_dict.items():
    stripped_keys[k.replace('module.','')] = v

model.load_state_dict(stripped_keys)

atten_dict = {}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for j, (img, label, idx) in enumerate(test_dataloader):
    with torch.no_grad():
        img, label = img.to(device), label.to(device)
        attentions = model.get_last_selfattention(img.clone())
        new_heatmaps = attentions[:,:,0,1:]
        avg_attn_maps = torch.mean(new_heatmaps, dim=1)
        for i in range(len(idx)):
            atten_dict[idx[i].item()] = avg_attn_maps[i,:].detach().cpu().numpy().tolist() # can't have idx as Tensor
    print(f'Processed batch {j}.')

    

with open("avg_attns_testset.json", "w") as outfile:
        json.dump(atten_dict, outfile)