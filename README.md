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
conda activate speml
```
or 
```bash
source activate spemlcpu
```