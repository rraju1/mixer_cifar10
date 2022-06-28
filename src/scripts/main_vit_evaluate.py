import argparse

import torch
import wandb
# wandb.login()

import matplotlib.pyplot as plt
from utils.dataloader import get_dataloaders
from utils.utils import get_model
from utils.train import Trainer_Masked_ViT
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.title(title)



def evaluate_model(model, test_dataloader, path, args):
    model.load_state_dict(torch.load(path))
    model.eval()
    model.set_phase('test')

    with torch.no_grad():
        true, preds = [], []
        corr = 0
        num_imgs = 0
        for img, lbl, idx in test_dataloader:
            true.append(lbl)
            img, lbl = img.to(args.device), lbl.to(args.device)
            inter_out = model(img)
            out = model.head(inter_out)
            pred = out.argmax(dim=-1)
            corr += pred.eq(lbl).sum(-1)
            preds.append(pred.cpu())
            num_imgs += img.size(0)
    return torch.cat(true).numpy(), torch.cat(preds).numpy(), corr, num_imgs





parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['c10', 'c100', 'svhn', 'vww'])
parser.add_argument('--model', required=True, choices=['mlp_mixer', 'mlp_mixer_masked', 'vit_small', 'vit_small_masked'])
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
parser.add_argument('--lambda-drop', type=float, default=0.1)
parser.add_argument('--attn_maps_path', type=str, default='./utils/avg_attns_vww_trainset.json')
parser.add_argument('--attn_maps_test_path', type=str, default='./utils/avg_attns_vww_testset.json')
parser.add_argument('--split', type=str, default='index')
parser.add_argument('--phase', type=str,default='train')
parser.add_argument('--patchdroptest', action='store_true')

# type
parser.add_argument('--exp_type', type=str, default='baseline')

args = parser.parse_args()
args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.nesterov = not args.off_nesterov
torch.random.manual_seed(args.seed)

if args.qk_scale == 'None':
    args.qk_scale = None

experiment_name = f"{args.model}_{args.dataset}_{args.optimizer}_{args.scheduler}"
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

experiment_name += f"baseline_test"


if __name__=='__main__':
    train_dl, test_dl = get_dataloaders(args)
    model = get_model(args)

    path = 'artifacts/vit_small_masked_vww_adam_cosine_salient_lambda0.5_masked.pth'

    y_true, y_pred, epoch_corr, num_imgs = evaluate_model(model, test_dl, path, args)
    print(f'corr: {epoch_corr} num_imgs: {num_imgs} acc: {epoch_corr/num_imgs}')
    plot_cm(y_true, y_pred, 'salient')
    plt.savefig('salient.png')


    # model.load_state_dict(torch.load(path))
    # model.set_phase('test')
    # epoch_corr, num_imgs = 0.0, 0.0
    

    # testcase = torch.ones(1,3,224,224).to(args.device)
    # out = model(testcase)
    # out = model.head(out)
    # print(f'testcase output: {out}')
    # print(model.state_dict())
    # exit()

    # for batch in test_dl:
    #     num_imgs += batch[0].size(0)
    #     model.eval()
    #     img, label, index = batch
    #     img, label = img.to(args.device), label.to(args.device)
        
    #     with torch.no_grad():
    #         out = model(img)
    #         out = model.head(out)
        
    #     epoch_corr += out.argmax(dim=-1).eq(label).sum(-1)
                
    # with wandb.init(project='vit', config=args, name=experiment_name):
    #     train_dl, test_dl = get_dataloaders(args)
    #     model = get_model(args)
    #     model.load_state_dict(torch.load('vit_baseline_epoch2.pth'))
    #     model.set_phase('test')
    #     trainer = Trainer_Masked_ViT(model, args)
    #     for batch in test_dl:
    #         trainer._test_one_step(batch)
    #     print(f'correct: {(trainer.epoch_corr)/36101}')

        # trainer.fit(train_dl, test_dl)
        # wandb.alert(
        #     title=f"{experiment_name}",
        #     text="Run finished."
        # )
        # torch.save(model.state_dict(), f'./{experiment_name}.pth')