# Getting Started with *HydroDL2*

Guide for beginning development with differentiable hydrology models in *DeltaMod* (`generic_deltamodel`).


## 1. System Requirements

*DeltaMod* uses PyTorch models requiring CUDA support only available with NVIDIA GPUs. Therefore, use of *DeltaMod* requires a system running 
- Windows or Linux
- NVIDIA GPU(s) supporting CUDA (>12.0 recommended)


## 2. Steps for Setup

For a functioning *DeltaMod* + *HydroDL2* build, 


### Clone the Repositories
- Open a terminal on your system, navigate to the directory where *DeltaMod* and *HydroDL2* will be stored, and clone:
  
    ```shell
    git clone https://github.com/mhpi/generic_diffModel.git
    git clone https://github.com/mhpi/hydroDL2.git
    ```
- Your install directory should now look like:

    .
    ├── generic_deltaModel/
    └── hydroDL2/ 


### Install the ENV
- A minimal package list is included with *DeltaMod* for getting started with differentiable models: `generic_deltaModel/envs/deltamod_env.yaml`.
- To install, run the following (optionally, include the `--prefix` flag to specify where you want the env downloaded):
     ```shell
     conda env create --file /generic_deltaModel/envs/deltamod_env.yaml
     ```
     or
  
     ```shell
     conda env create --prefix path/to/env --file /generic_deltaModel/envs/deltamod_env.yaml
     ```
- Activate the env with `conda activate deltamod` and open a python instance to check that CUDA is available with PyTorch:
     ```python
     import torch
     print(torch.cuda.is_available())
     ```
- If CUDA is not available, uninstall PyTorch from the env and reinstall according to your system specifications [here](https://pytorch.org/get-started/locally/). E.g.,
     ```shell
     conda uninstall pytorch
     conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
     ```


### Install *HydroDL2*
- To have the *HydroDL2* package accessible within *DeltaMod*, install with pip like so (optionally, include the `-e` flag to install with (hatch's) developer mode):
     ```shell
     cd hydroDL2
     pip install .
     ```
     or
  
     ```shell
     cd hydroDL2
     pip install -e .
     ```

---