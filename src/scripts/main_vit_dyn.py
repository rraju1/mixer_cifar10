import argparse

import torch
import wandb
wandb.login()

from utils.dataloader import get_dataloaders
from utils.utils import get_model
from utils.train import Trainer_DViT

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['c10', 'c100', 'svhn', 'vww'])
parser.add_argument('--model', required=True, choices=['mlp_mixer', 'mlp_mixer_masked', 'vit_small', 'vit_small_masked', 'vit_dyn'])
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--eval-batch-size', type=int, default=32)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--size', type=int, default=96)
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--root', type=str, default='/group/ece/ececompeng/lipasti/libraries/datasets/vw_coco2014_96')
# parser.add_argument('--precision', type=int, default=16)

parser.add_argument('--patch-size', type=int, default=16)
parser.add_argument('--hidden-size', type=int, default=192) # embed_dim
parser.add_argument('--hidden-c', type=int, default=512)
parser.add_argument('--hidden-s', type=int, default=64)
parser.add_argument('--num-layers', type=int, default=8) # depth
parser.add_argument('--drop-p', type=int, default=0.)
parser.add_argument('--off-act', action='store_true', help='Disable activation function')
parser.add_argument('--is-cls-token', action='store_true', help='Introduce a class token.')

# vit arguments
parser.add_argument('--in_chans', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=12)
parser.add_argument('--mlp_ratio', type=int, default=4)
parser.add_argument('--qkv_bias', action='store_false')
parser.add_argument('--qk_scale', type=str, default='None')
parser.add_argument('--attn_drop_rate', type=float, default=0.)
parser.add_argument('--drop_path_rate', type=float, default=0.)

# Optimizer/Scheduler params
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--min-lr', type=float, default=1e-6)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
parser.add_argument('--scheduler', default='cosine', choices=['step', 'cosine'])
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--weight-decay', type=float, default=5e-5)
parser.add_argument('--off-nesterov', action='store_true')
parser.add_argument('--label-smoothing', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--warmup-epoch', type=int, default=5)

# augmentations
parser.add_argument('--autoaugment', action='store_true')
parser.add_argument('--clip-grad', type=float, default=0, help="0 means disabling clip-grad")
parser.add_argument('--cutmix-beta', type=float, default=1.0)
parser.add_argument('--cutmix-prob', type=float, default=0.)
parser.add_argument('--padding', type=int, default=4)


## Patch drop args
parser.add_argument('--regularizer', type=float, default=0.1)
parser.add_argument('--lambda-drop', type=float, default=0.1)
parser.add_argument('--attn_maps_path', type=str, default='./utils/avg_attns_vww_trainset.json')
parser.add_argument('--attn_maps_test_path', type=str, default='./utils/avg_attns_vww_testset.json')
parser.add_argument('--split', type=str, default='index')
parser.add_argument('--phase', type=str,default='train')
parser.add_argument('--patchdroptest', action='store_true')

# type
parser.add_argument('--exp_type', type=str, default='dynamic')

args = parser.parse_args()
args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.nesterov = not args.off_nesterov
torch.random.manual_seed(args.seed)

if args.qk_scale == 'None':
    args.qk_scale = None

experiment_name = f"{args.exp_type}_{args.lambda_drop}"
if args.autoaugment:
    experiment_name += "_aa"
if args.clip_grad:
    experiment_name += f"_cg{args.clip_grad}"
if args.off_act:
    experiment_name += f"_noact"
if args.cutmix_prob>0.:
    experiment_name += f'_cm'
if args.is_cls_token:
    experiment_name += f"_cls"


if __name__=='__main__':
    with wandb.init(project='vit', config=args, name=experiment_name):
        train_dl, test_dl = get_dataloaders(args)
        model = get_model(args)
        trainer = Trainer_DViT(model, args)
        trainer.fit(train_dl, test_dl)
        wandb.alert(
            title=f"{experiment_name}",
            text="Dynamic run finished."
        )
        torch.save(model.state_dict(), f'./{experiment_name}.pth')