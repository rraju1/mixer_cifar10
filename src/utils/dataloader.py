import torch
import torchvision
import torchvision.transforms as transforms
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from PIL import Image
from torch.utils.data import Dataset
from utils.autoaugment import CIFAR10Policy, SVHNPolicy


class IndexedDataset(Dataset):
    def __init__(self, data, labels, transform, split):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.split = split
    def __getitem__(self, index):
        x = self.data[index]
        x = Image.fromarray(x)
        y = self.labels[index]
        if self.transform:
            x = self.transform(x)
        if self.split == "index":
            return x, y, index
        return x, y
    def __len__(self):
        return len(self.data)


def get_dataloaders(args):
    train_transform, test_transform = get_transform(args)

    if args.dataset == "c10":
        train_ds = torchvision.datasets.CIFAR10('./datasets', train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10('./datasets', train=False, transform=test_transform, download=True)
        train_ds = IndexedDataset(train_ds.data, train_ds.targets, train_transform, args.split)
        test_ds = IndexedDataset(test_ds.data, test_ds.targets, test_transform, args.split)
    elif args.dataset == "c100":
        train_ds = torchvision.datasets.CIFAR100('./datasets', train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100('./datasets', train=False, transform=test_transform, download=True)
    elif args.dataset == "svhn":
        train_ds = torchvision.datasets.SVHN('./datasets', split='train', transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN('./datasets', split='test', transform=test_transform, download=True)
    elif args.dataset == "vww":
        train_ds = VisualWakeWordsDataset(args.root, split='train', transform=train_transform, idx=args.split=='index')
        test_ds = VisualWakeWordsDataset(args.root, split='val', transform=test_transform, idx=args.split=='index')
    else:
        raise ValueError(f"No such dataset:{args.dataset}")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_dl, test_dl

def get_transform(args):
    if args.dataset in ["c10", "c100", 'svhn', 'vww']:
        if args.dataset=="c10":
            args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        elif args.dataset=="c100":
            args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        elif args.dataset=="svhn":
            args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        elif args.dataset=="vww":
            args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # just use imagenet mean,std
    else:
        args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_transform_list = [transforms.Resize(args.size),
        transforms.RandomCrop(size=(args.size,args.size), padding=args.padding)
    
    ]

    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform_list.append(CIFAR10Policy())
        elif args.dataset == 'svhn':
            train_transform_list.append(SVHNPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")   
        

    train_transform = transforms.Compose(
        train_transform_list+[
            transforms.ToTensor(),
            transforms.Normalize(
                mean=args.mean,
                std = args.std
            )
        ]
    )
    test_transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=args.mean,
            std = args.std
        )
    ])

    return train_transform, test_transform


class VisualWakeWordsDataset(VisionDataset):
    """`Visual Wake Words <https://arxiv.org/abs/1906.05721>`_ Dataset.
    Args:
        root (string): Root directory where COCO images are downloaded to.
        split (string): train and val split
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, split, transform=None, target_transform=None, transforms=None, idx=False):
        super(VisualWakeWordsDataset, self).__init__(root, transforms, transform, target_transform)
        self.idx = idx
        self.split = split
        self.img_label_list = self.process_txt_file(self.root, self.split)
    
    def process_txt_file(self, root, split):
        textfile = os.path.join(root, split + '.txt')
        with open(textfile) as f:
            mylist = f.read().splitlines()
        for idx in range(len(mylist)):
            mylist[idx] = mylist[idx].split()
        return mylist
    
    def __getitem__(self, index):
        img, label = self.img_label_list[index][0], int(self.img_label_list[index][1])
        img_path = os.path.join(self.root, self.split, img)
        img = default_loader(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        
        if self.idx:
            return img, label, index
        return img, label

    def __len__(self):
        return len(self.img_label_list)