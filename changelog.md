# change log

Changed MixerLayer_mask back to MixerLayer since enforcing mask in intermediate layers could be reason for training instability. => helped on patchdrop 20% but diverged for higher pruning rates.

Maybe we can take the gradients of the original images and occuluded ones and compare their cosine stiffness to make sure they are aligned. => For the most part gradients are aligned. 

Changed architecture to ViT. Tested out different patch sizes and batch sizes. Seems like it converges but weird relationship with odd and even patchdrop rates.