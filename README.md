# Robust Image Recovery via Double Over-parameterization

This repository is the official implementation of the paper [Robust Recovery via Implicit Bias of Discrepant Learning Rates for Double Over-parameterization](https://arxiv.org/abs/2006.08857) by Chong You (UC Berkeley), Zhihui Zhu (University of Denver), Qing Qu (NYU) and Yi Ma (UC Berkeley). 

Given an image corrupted with salt-and-pepper noise (left), our method produces an image (middle) that closely recovers the underlying clean image (right).

![image](https://github.com/ChongYou/robust-image-recovery/blob/master/figs/corrupted.jpg)    ![image](https://github.com/ChongYou/robust-image-recovery/blob/master/figs/recovered.jpg)    ![image](https://github.com/ChongYou/robust-image-recovery/blob/master/figs/clean.jpg)

The following learning curves show that our method produces higher PSNR (Ours (width=192) below) and does not overfit when compared to previous Deep Image Prior method (DIP-L1 (width=192) below).

![image](https://github.com/ChongYou/robust-image-recovery/blob/master/figs/PSNR.png)

# Image Recovery for different test images (Figure 3)

The following commands perform image recovery for images in the folder data/denoising/. Artificial salt-and-pepper noise that corrupts 50% of the pixels is injected to the images for testing purposes.

```
python denoising.py --gpu 0 --ckpt F16_05_128 --image F16 --nlevel 0.5 --width 128
python denoising.py --gpu 0 --ckpt Peppers_05_128 --image Peppers --nlevel 0.5 --width 128
python denoising.py --gpu 0 --ckpt Baboon_05_128 --image Baboon --nlevel 0.5 --width 128 
python denoising.py --gpu 0 --ckpt kodim03_05_128 --image kodim03 --nlevel 0.5 --width 128 
```
Recovered images are found in the folder checkpoints/ with file names

F16_05_128_last.png, peppers_05_128_last.png, baboon_05_128_last.png, kodim03_05_128_last.png

These images reproduce Figure 3 of the paper. The associated learning curves are in Figure 5 (left). 

# Image Recovery for varying corruption levels (Figure 4)

The following commands perform image recovery for the image in data/denoising/F16. Artificial salt-and-pepper noise that corrupts {10%, 30%, 50%, 70%} of the pixels is injected to the image for testing purposes.

```
python denoising.py --gpu 0 --ckpt F16_01_128 --image F16 --nlevel 0.1 --width 128
python denoising.py --gpu 0 --ckpt F16_03_128 --image F16 --nlevel 0.3 --width 128
python denoising.py --gpu 0 --ckpt F16_05_128 --image F16 --nlevel 0.5 --width 128
python denoising.py --gpu 0 --ckpt F16_07_128 --image F16 --nlevel 0.7 --width 128
```
Recovered images are found in the folder checkpoints/ with file names

F16_01_128_last.png, F16_03_128_last.png, F16_05_128_last.png, F16_07_128_last.png

These images reproduce Figure 4 of the paper. The associated learning curves are in Figure 5 (right). 

# Installation / dependencies

```
conda env create -f environment.yml
conda activate double-overparam
```

# Citing this work

```
@article{you2020robust,
  title={Robust Recovery via Implicit Bias of Discrepant Learning Rates for Double Over-parameterization},
  author={You, Chong and Zhu, Zhihui and Qu, Qing and Ma, Yi},
  journal={arXiv preprint arXiv:2006.08857},
  year={2020}
}
```
