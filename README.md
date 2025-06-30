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