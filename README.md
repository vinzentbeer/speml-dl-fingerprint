# Initizializing the conda environment

Two ways to init either with cuda support or wihtout cuda support. Run the following command in the terminal:

## With cuda support
```bash
conda env create -f environment.yml
```
## Without cuda support
```bash
conda env create -f environment_cpu.yml
```

## Activate the environment
```bash
conda activate spemlGPU
```
Add specific pytorch version according to your cuda version, for example:
https://pytorch.org/get-started/locally/
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
or 
```bash
conda activate spemlcpu
```


# TODOs for the assignment

- [x] Train squeezeNet on MNIST and FashionMNIST and export the model
- [ ] Model acrchitecture:
  - [ ] Maybe change the first layer to have a correct input size (MNIST is 1 channel, FashionMNIST is 1 channel, but SqueezeNet expects 3 channels)
- [ ] Baseline: Train for more episodes. Fix the first n layers and don't update them (keeps edge detection)
- [ ] Embed fingerprint in the model
  - [x] Update the watermarking method to create batches with normal dataset + watermark image! (Keeps accuracy  high)
  - [x] USING PRETRAINED METHOD (Train baseline model first, then use the pretrained model to embed the watermark)
  - [ ] Check transformer of image. Currently using imagenet transformer (Transforms size, color normalization of the image)
- [x] Attack the watermarked model
  - [x] Update the attack using the original dataset
- [x] Evaluate the model with and without watermarked


# References to other repos used in this project

The corresponding licenses of these repositories are in their respective folders.

- [SqueezeNet](https://github.com/forresti/SqueezeNet) see `SqueezeNet/LICENSE`
- [WatermarkNN](https://github.com/adiyoss/WatermarkNN) see `WatermarkNN/LICENSE`