import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import warmup_scheduler
import pickle as pkl
import os

from types import SimpleNamespace
from copy import deepcopy
from utils.dataloader import get_dataloaders
from utils.utils import rand_bbox, get_model, compute_cos, gradient_mag_sim

def train_model(model, train_dataloader, optimizer, criterion, scheduler, scaler, args, num_epochs):
    model.set_lambda(0.0) # set to no patches being dropped
    print(f'Setting lambda to {model.get_lambda()}')
    for epoch in range(num_epochs):
        # training part of loop
        print(f'Training epoch {epoch}')
        for img, label, idx in train_dataloader:
            img, label = img.to(args.device), label.to(args.device)
            optimizer.zero_grad()
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(img.size(0)).to(args.device)
                target_a = label
                target_b = label[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
                img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
                # compute output
                out = model(img, idx)
                out = model.head(out)
                loss = criterion(out, target_a) * lam + criterion(out, target_b) * (1. - lam)
                # with torch.cuda.amp.autocast():
                #     out = model(img, idx)
                #     out = model.head(out)
                #     loss = criterion(out, target_a) * lam + criterion(out, target_b) * (1. - lam)
            else:
                # compute output
                out = model(img, idx)
                out = model.head(out)
                loss = criterion(out, label)
                # with torch.cuda.amp.autocast():
                #     out = model(img, idx)
                #     out = model.head(out)
                #     loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
        scheduler.step()
        savepath = f"{os.getcwd()}/stiffness_ckpts/torch_model_{epoch}.pth"
        torch.save(model.state_dict(), savepath)
        print(f"Finished training epoch {epoch}. Saving ckpt to " + savepath)

def evaluate_sims(model, lambda1, epoch, train_dataloader, args, optimizer, criterion, scaler):
    model.train()
    savepath = f"{os.getcwd()}/stiffness_ckpts/torch_model_{epoch}.pth"
    model.load_state_dict(torch.load(savepath))
    cos_sim_list = []
    grad_sim_list = []
    for img, label, idx in train_dataloader:
        img, label = img.to(args.device), label.to(args.device)
        r = np.random.rand(1)
        if args.cutmix_beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
            rand_index = torch.randperm(img.size(0)).to(args.device)
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            # compute output
            model.set_lambda(0.0)
            out = model(img, idx)
            out = model.head(out)
            loss = criterion(out, target_a) * lam + criterion(out, target_b) * (1. - lam)
            loss.backward()
            parameters = deepcopy([p.grad.data.detach().cpu() for p in model.parameters() if p.grad is not None and p.requires_grad])
            optimizer.zero_grad()
            model.set_lambda(lambda1)
            out = model(img, idx)
            out = model.head(out)
            loss = criterion(out, target_a) * lam + criterion(out, target_b) * (1. - lam)
            loss.backward()
            occ_parameters = deepcopy([p.grad.data.detach().cpu() for p in model.parameters() if p.grad is not None and p.requires_grad])
            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():
            #     model.set_lambda(0.0)
            #     scaler.scale(loss).backward()
            #     parameters = deepcopy([p.grad.data.detach().cpu() for p in model.parameters() if p.grad is not None and p.requires_grad])
            #     optimizer.zero_grad()
            #     model.set_lambda(lambda1)
            #     out = model(img, idx)
            #     out = model.head(out)
            #     loss = criterion(out, target_a) * lam + criterion(out, target_b) * (1. - lam)
            #     scaler.scale(loss).backward()
            #     occ_parameters = deepcopy([p.grad.data.detach().cpu() for p in model.parameters() if p.grad is not None and p.requires_grad])
            #     optimizer.zero_grad()
        else:
            # compute output
            model.set_lambda(0.0)
            out = model(img, idx)
            out = model.head(out)
            loss = criterion(out, label)
            loss.backward()
            parameters = deepcopy([p.grad.data.detach().cpu() for p in model.parameters() if p.grad is not None and p.requires_grad])
            optimizer.zero_grad()
            model.set_lambda(lambda1)
            out = model(img, idx)
            out = model.head(out)
            loss = criterion(out, label)
            loss.backward()
            occ_parameters = deepcopy([p.grad.data.detach().cpu() for p in model.parameters() if p.grad is not None and p.requires_grad])
            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():
            #     model.set_lambda(0.0)
            #     out = model(img, idx)
            #     out = model.head(out)
            #     loss = criterion(out, label)
            #     scaler.scale(loss).backward()
            #     parameters = deepcopy([p.grad.data.detach().cpu() for p in model.parameters() if p.grad is not None and p.requires_grad])
            #     optimizer.zero_grad()
            #     model.set_lambda(lambda1)
            #     out = model(img, idx)
            #     out = model.head(out)
            #     loss = criterion(out, label)
            #     scaler.scale(loss).backward()
            #     occ_parameters = deepcopy([p.grad.data.detach().cpu() for p in model.parameters() if p.grad is not None and p.requires_grad])
            #     optimizer.zero_grad()
        cur_cos_sim = compute_cos(parameters, occ_parameters)
        cur_grad_sim = gradient_mag_sim(parameters, occ_parameters)
        cos_sim_list.append(cur_cos_sim)
        grad_sim_list.append(cur_grad_sim)

    cos_sum = sum(cos_sim_list)
    total_len = len(cos_sim_list)
    print(f'cos sum: {cos_sum}, total_len {total_len}')
    avg_cos = cos_sum/total_len
    print(f'avg cos sum: {avg_cos}')

    grad_sum = sum(grad_sim_list)
    total_len = len(grad_sim_list)
    print(f'grad sum: {grad_sum}, total_len {total_len}')
    avg_grad = grad_sum/total_len
    print(f'avg grad sum: {avg_grad}')

    print(f'Finished processing for cos stiffness and grad sim')
    return avg_cos, avg_grad


def main():
    args = SimpleNamespace()
    args.dataset = 'vww'
    args.root = '/group/ece/ececompeng/lipasti/libraries/datasets/vw_coco2014_96'
    args.model = 'vit_small_masked'
    args.batch_size = 128
    args.eval_batch_size = 10
    args.num_workers = 4
    args.seed = 0
    args.epochs = 100
    args.patch_size = 16
    args.autoaugment = False
    args.use_cuda = True
    args.size = 224
    args.split = 'index'
    args.in_chans = 3
    args.hidden_size = 384
    args.num_layers = 8
    args.mlp_ratio = 4
    args.qkv_bias = True
    args.num_classes = 2
    args.qk_scale = None
    args.drop_p = 0
    args.attn_drop_rate = 0
    args.drop_path_rate = 0
    args.num_heads = 8
    args.attn_maps_path = './utils/avg_attns_vww_trainset.json'
    args.attn_maps_test_path = './utils/avg_attns_vww_testset.json'
    args.phase = 'train'
    args.patchdroptest = False
    args.padding = 4
    args.lambda_drop = 0.3
    args.momentum = 0.9
    args.weight_decay = 5e-5
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.nesterov = True
    args.beta1 = 0.9
    args.beta2 = 0.99
    args.warmup_epoch = 5
    args.lr = 1e-3
    args.min_lr = 1e-6
    args.label_smoothing = 0.1
    args.cutmix_beta = 1
    args.cutmix_prob = 0.5

    train_dataloader, test_dataloader = get_dataloaders(args)

    model = get_model(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=base_scheduler)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()
    scaler = torch.cuda.amp.GradScaler()

    # saved in stiffness_ckpts
    num_epochs = 11
    train_model(model, train_dataloader, optimizer, criterion, scheduler, scaler, args, num_epochs)

    epoch_cos_sim = {}
    epoch_grad_sim = {}

    for lambda1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        cos_list = []
        grad_list = []
        for epoch in range(num_epochs):
            print(f'Patchdrop with lambda {lambda1} and epoch {epoch}')
            avg_cos, avg_grad = evaluate_sims(model, lambda1, epoch, train_dataloader, args, optimizer, criterion, scaler)
            cos_list.append(avg_cos)
            grad_list.append(avg_grad)
        epoch_cos_sim[lambda1] = cos_list
        epoch_grad_sim[lambda1] = grad_list

    with open(f'epoch_cos_sim_just.pkl', 'wb') as input_file:
        pkl.dump(epoch_cos_sim , input_file)

    with open(f'epoch_grad_sim.pkl', 'wb') as input_file:
        pkl.dump(epoch_grad_sim , input_file)


if __name__ == "__main__":
    main()