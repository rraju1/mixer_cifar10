import unittest
import torch
import numpy as np
from types import SimpleNamespace
from utils.dataloader import get_dataloaders
from utils.utils import project_bin_mask
from src.utils.mlp_mixer import MixerLayer_mask, create_binary_mask, MLP2_mask, \
                          MLP1, MLP1_mask, MaskedMLPMixer
import json




class TestMaskedMLPMixer(unittest.TestCase):
    def setUp(self):
        self.testmixer = MaskedMLPMixer(3, 224, 16, 512)
    
    def test_MLP_masks(self):
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
        
        # input = x * bin_mask
        output, mlp_outputs = self.testmixer(image, masks)
        mlp_masks = create_binary_mask(mlp_outputs)
        norm = np.linalg.norm(mlp_masks.detach().numpy() - masks.detach().numpy())
        self.assertLess(norm,1)


class TestMixerLayer(unittest.TestCase):
    def setUp(self):
        self.testmixer = MixerLayer_mask(2, 100, 100, 100, 0, False)
    
    def test_MLP_masks(self):
        x = torch.randn(2,100)
        bin_mask = torch.vstack(
            (torch.ones(100) , torch.zeros(100))
        
        )
        input = x * bin_mask
        output = self.testmixer(input)
        norm = np.linalg.norm(output[1,:].detach().numpy() - input[1,:].detach().numpy())
        self.assertLess(norm,1)


class TestMLP2(unittest.TestCase):
    def setUp(self):
        self.testmlp2 = MLP2_mask(100, 100, 0, False)
    
    def test_MLP_masks(self):
        x = torch.randn(2,100)
        bin_mask = torch.vstack(
            (torch.ones(100) , torch.zeros(100))
        
        )
        input = x * bin_mask
        output = self.testmlp2(input, bin_mask)
        norm = np.linalg.norm(output[1,:].detach().numpy() - input[1,:].detach().numpy())
        self.assertLess(norm,1)

class TestMLP1(unittest.TestCase):
    def setUp(self):
        self.testmlp = MLP1(14, 100, 100, 0.0, False)
        self.testmlp1 = MLP1_mask(2, 100, 100, 0.0, False)
    
    def test_mlp_shape(self):
        x = torch.randn(14,100)
        output = self.testmlp(x)
        self.assertEqual(output.shape, x.shape)

    def test_LN_shape(self):
        x = torch.randn(14,100)
        output = self.testmlp.ln(x)
        self.assertEqual(output.shape, x.shape)

    def test_MLP_masks(self):
        x = torch.randn(2,100)
        bin_mask = torch.vstack(
            (torch.ones(100) , torch.zeros(100))
        
        )
        input = x * bin_mask
        output = self.testmlp1(input, bin_mask)
        norm = np.linalg.norm(output[1,:].detach().numpy() - input[1,:].detach().numpy())
        self.assertLess(norm,1)



if __name__ == "__main__":
    unittest.main()