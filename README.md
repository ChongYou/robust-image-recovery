# Robust Recovery via Double Over-parameterization

This repository is the official implementation of [Robust Recovery via Implicit Bias of Discrepant Learning Rates
for Double Over-parameterization]

# Install

```
conda env create -f environment.yml
conda activate double-overparam
```

# Image Recovery for different test images (Figure 3)

Run the following commands:

```
python denoising.py --gpu 0 --ckpt F16_05_128 --image F16 --nlevel 0.5 --width 128
python denoising.py --gpu 0 --ckpt Peppers_05_128 --image Peppers --nlevel 0.5 --width 128
python denoising.py --gpu 0 --ckpt Baboon_05_128 --image Baboon --nlevel 0.5 --width 128 
python denoising.py --gpu 0 --ckpt kodim03_05_128 --image kodim03 --nlevel 0.5 --width 128 
```
which will print the PSNR as learning proceeds. Recovered images are found in the folder checkpoints/ with file names

F16_05_128_last.png, peppers_05_128_last.png, baboon_05_128_last.png, kodim03_05_128_last.png

# Image Recovery for varying corruption levels (Figure 4)

Run the following commands:

```
python denoising.py --gpu 0 --ckpt F16_01_128 --image F16 --nlevel 0.1 --width 128
python denoising.py --gpu 0 --ckpt F16_03_128 --image F16 --nlevel 0.3 --width 128
python denoising.py --gpu 0 --ckpt F16_05_128 --image F16 --nlevel 0.5 --width 128
python denoising.py --gpu 0 --ckpt F16_07_128 --image F16 --nlevel 0.7 --width 128
```
which will print the PSNR as learning proceeds. Recovered images are found in the folder checkpoints/ with file names

F16_01_128_last.png, F16_03_128_last.png, F16_05_128_last.png, F16_07_128_last.png
