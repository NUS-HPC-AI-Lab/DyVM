# Semantic Segmentation

## Environment

We adapt from the Vim codebase to conduct the semantic segmentation task. All scripts are provided in the `seg/scripts/` directory. The environment is the same as the Vim codebase, except we use our own `mmsegmentation` codebase to add our token ratio and block ratio logic.

```bash
# create conda environment
conda create -n your_env_name python=3.9.19
pip3 install numpy==1.23.4

# install torch
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# install mmseg components
pip install -U openmim
mim install mmcv-full==1.7.1
pip3 install -e mmsegmentation
```

Finally, change `data_root` in `seg/configs/_base_/datasets/ade20k.py` to the path of your ADE20K dataset.

## Quick Start

We provide all training scripts used in our experiments in `seg/scripts/`. Simply run the script to train your model.

```bash
cd seg
bash scripts/ft_vim_tiny_upernet.sh
```

