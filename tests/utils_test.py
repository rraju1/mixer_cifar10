import unittest
import torch
from copy import deepcopy
from src.utils.mlp_mixer import create_binary_mask
from src.utils.dataloader import get_dataloaders
from src.utils.utils import project_bin_mask
import json
from types import SimpleNamespace
import math



class TestUtils(unittest.TestCase):

    """
    Given tensor x, create a binary masks which preserves the positions of the zeros
    """
    def test_binary_mask(self):
        x = torch.randn(3,3)
        y = deepcopy(x)
        x[x < 0.5] = 0
        mask = create_binary_mask(x)
        self.assertEqual(torch.sum(y * mask), torch.sum(x))

    def test_project_mask(self):
        with open('./src/utils/avg_attns_trainset.json') as json_file:
            data = json.load(json_file)

        args = SimpleNamespace()
        args.dataset = 'c10'
        args.model = 'mlp_mixer'
        args.batch_size = 10
        args.eval_batch_size = 10
        args.num_workers = 4
        args.seed = 0
        args.epochs = 300
        args.patch_size = 16
        args.autoaugment = False
        args.use_cuda = False
        args.size = 224
        args.split = 'index'
        args.lambda_drop = 0.5
        args.hidden_size = 512
        train_dataloader, test_dataloader = get_dataloaders(args)
        image, label, index = next(iter(train_dataloader))
        proj_mask = project_bin_mask(image, index, data, args.lambda_drop, args.patch_size, 3, args.hidden_size)
        masks = create_binary_mask(proj_mask)
        for j in range(args.batch_size):
            with self.subTest(j=j):
                self.assertEqual(torch.sum(masks[j,:]).item(), round(((1 - args.lambda_drop) * 196)) * args.hidden_size)


if __name__ == "__main__":
    unittest.main()