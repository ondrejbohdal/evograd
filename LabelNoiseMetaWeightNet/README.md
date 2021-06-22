# Meta-Weight-Net with EvoGrad - Noisy Labels
Our implementation extends the original implementation from Shu et al. for the NeurIPS'19 paper Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting (Official Pytorch implementation for class-imbalance).


## Running Meta-Weight-Net with EvoGrad on benchmark datasets (CIFAR-10 and CIFAR-100)
Example:
```
python meta-weight-net-label-noise.py --dataset cifar10 --corruption_type flip2 --corruption_prob 0.4 --method evo
```

## Requirements
- PyTorch
- higher
- tqdm


## Further information about Meta-Weight-Net
NeurIPS'19: Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting (Official Pytorch implementation for class-imbalance).
[Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting](https://arxiv.org/abs/1902.07379)  
Jun Shu, Qi Xie, Lixuan Yi, Qian Zhao, Sanping Zhou, Zongben Xu, Deyu Meng
The original implementation of noisy labels is available at https://github.com/xjtushujun/Meta-weight-net.



