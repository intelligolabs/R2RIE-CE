
<h1 align="center">
Mind the Error! Detection and Localization of Instruction Errors in Vision-and-Language Navigation
</h1>

<div>
    <p align="center">
    <a href='https://francescotaioli.github.io/' target='_blank'>Francesco Taioli</a>;
    Stefano Rosa;
     Alberto Castellini, Lorenzo Natale, Alessio Del Bue, Alessandro Farinelli, Marco Cristani, Yiming Wang
    </p>
</div>

<h3 align="center">
<!-- <a href="https://arxiv.org/abs/2303.00304">Paper</a> | -->
 <!-- <a href="https://youtu.be/oLo3L0oMcWQ">Video</a> | -->
 Accepted to
  <a href="https://iros2024-abudhabi.org/">IROS 24</a></h3>

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2403.10700-b31b1b.svg)](https://arxiv.org/abs/2403.10700) -->

<div align="center">
  <strong><a href="https://intelligolabs.github.io/R2RIE-CE/">Project Page (Paper, Code and Dataset)</a></strong>
</div>


<p align="center">
contact: <code>francesco.taioli@polito.it</code>
</p>
<hr>

> [!IMPORTANT]
> Consider citing our paper:
> ```BibTeX
>   @article{taioli2024mind,
>   title={{Mind the error! detection and localization of instruction errors in vision-and-language navigation}},
>   author={Taioli, Francesco and Rosa, Stefano and Castellini, Alberto and Natale, Lorenzo and Del Bue, Alessio and Farinelli, Alessandro and Cristani, Marco and Wang, Yiming},
>   journal={arXiv preprint arXiv:2403.10700},
>   year={2024}
>   }
>   ```


## Abstract
Vision-and-Language Navigation in Continuous Environments (VLN-CE) is one of the most intuitive yet challenging embodied AI tasks. Agents are tasked to navigate towards a target goal by executing a set of low-level actions, following a series of natural language instructions. All VLN-CE methods in the literature assume that language instructions are exact. However, in practice, instructions given by humans can contain errors when describing a spatial environment due to inaccurate memory or confusion. Current VLN-CE benchmarks do not address this scenario, making the state-of-the-art methods in VLN-CE fragile in the presence of erroneous instructions from human users. For the first time, we propose a novel benchmark dataset that introduces various types of instruction errors considering potential human causes. This benchmark provides valuable insight into the robustness of VLN systems in continuous environments. We observe a noticeable performance drop (up to -25%) in Success Rate when evaluating the state-of-the-art VLN-CE methods on our benchmark. Moreover, we formally define the task of Instruction Error Detection and Localization, and establish an evaluation protocol on top of our benchmark dataset. We also propose an effective method, based on a cross-modal transformer architecture, that achieves the best performance in error detection and localization, compared to baselines. Surprisingly, our proposed method has revealed errors in the validation set of the two commonly used datasets for VLN-CE, i.e., R2R-CE and RxR-CE, demonstrating the utility of our technique in other tasks.

Table of contents
=================

<!--ts-->
   * [Setup](#setup)
      * [Install dependencies](#install-dependencies)
      * [Download models and task dataset](#download-models-and-task-dataset)
   * [How to run](#how-to-run)
   * [Docs](#docs)
   * [Acknowledge](#acknowledge)

<!--te-->

## Setup

### Install dependencies
1. Create a virtual environment (tested with ```python 3.7```, ```torch 1.9.1+cu111```, ```torch-scatter 2.0.9+cu11```). and install base dependencies.
   ```bash
   conda create --name r2r_ie_ce python=3.7.12 -c conda-forge
   conda activate r2r_ie_ce
   ```
2. Download the Matterport3D scene meshes. `download_mp.py` must be obtained from the Matterport3D [project webpage](https://niessner.github.io/Matterport/).

   ```bash
   # run with python 2.7
   python download_mp.py --task habitat -o data/scene_datasets/mp3d/
   # Extract to: ./data/scene_datasets/mp3d/{scene}/{scene}.glb
   ```
Extract such that it has the form ```data/scene_datasets/mp3d/{scene}/{scene}.glb.``` There should be 90 scenes. Place the scene_datasets folder in ```data```

3. Follow the [Habitat Installation Guide](https://github.com/facebookresearch/habitat-lab#installation) to install [`habitat-sim`](https://github.com/facebookresearch/habitat-sim) and [`habitat-lab`](https://github.com/facebookresearch/habitat-lab). We use version [`v0.1.7`](https://github.com/facebookresearch/habitat-lab/releases/tag/v0.1.7) in our experiments. In brief:

- Install `habitat-sim` for a machine with multiple GPUs or without an attached display (i.e. a cluster):
   ```bash
   # option 1 - faster
   wget https://anaconda.org/aihabitat/habitat-sim/0.1.7/download/linux-64/habitat-sim-0.1.7-py3.7_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2
   conda install --use-local habitat-sim-0.1.7-py3.7_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2

   ```
   ```bash
   # option 2 - slower
   conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
   ```
4. Install our project dependencies:
   ```bash
   pip install --ignore-installed -r requirements.txt
   ```
5. Clone `habitat-lab` from the github repository and install. The command below will install the core of Habitat Lab as well as the habitat_baselines.

   ```bash
   git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git
   cd habitat-lab
   python setup.py develop --all # install habitat and habitat_baselines
   ```

6. Install the tested version of torch - ```torch==1.9.1+cu111``` and other dependencies:
   ```bash
   pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
   pip install tensorboard==1.15.0 #  TensorBoard logging requires TensorBoard version 1.15 or above
   ```

---

### Download models and task dataset
7. Download BEVBert weights ```ckpt.iter9600.pth``` [[link]](https://drive.google.com/file/d/1-2u1NWmwpX09Rg7uT5mABo-CBTsLthGm/view?usp=sharing) in `ckpt`folder. Can also be done with gdown (must be installed with ```pip install gdown```). This model is the best BEVBert model ckpts, **to be downloaded only if you want train IEDL from scratch**. Otherwise, you can skip this step and download *IEDL*
   ```python
   gdown --fuzzy [link]
   ```
8. Download IEDL (TODO)
   ```bash
   gdown --fuzzy [link]
   ```
9. Download the waypoint predictor ```check_cwp_bestdist_hfov90``` [[link]](https://drive.google.com/file/d/1goXbgLP2om9LsEQZ5XvB0UpGK4A5SGJC/view) for CE (continuous environment) and place it in ```data/wp_pred```
   ```bash
   gdown --fuzzy [link]
   ```

10. Download the ```task dataset - R2RIE-CE``` from gdrive, and place it under ``` data/datasets/```
   
      ```bash
      cd data/datasets
      gdown --fuzzy https://drive.google.com/file/d/1GbypzvkiQ-e8M2I77UU5YDIZXi1sHkC3/view?usp=sharing
      unzip R2RIE_CE_1_3_v1.zip; rm -rf R2RIE_CE_1_3_v1.zip
      ```

11. Download ```gibson-2plus-resnet50.pth``` [[link]](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-2plus-resnet50.pth) and place in a folder of your choice. 
      ```bash
      wget [link]
      ```
Then, set the path of this ```.pth``` in ```MODEL.DEPTH_ENCODER.ddppo_checkpoint``` in ```eval``` and ```train``` scripts.


## How to run
For training:
Go to ```run_R2RIE-CE/train.bash``` and set a folder name to save your checkpoints. To do that, set the variale ```WANDB_RUN_NAME```. Then, copy the original BEVBert ckpt - ```ckpt/ckpt.iter9600.pth``` - in that folder and run the following command:
```bash
CUDA_VISIBLE_DEVICES="0,1" bash run_R2RIE-CE/train.bash 2333
```

For evaluation:
```bash
CUDA_VISIBLE_DEVICES="0,1" bash run_R2RIE-CE/eval.bash 2333
```

## Docs
See the documentation on how to use the dataset (changing sensor, update task definition, ecc) in the [docs](docs/docs.md) folder.

# Acknowledge

Our implementation is inspired by [BEVBert](https://github.com/MarSaKi/VLN-BEVBert).

Thanks for open sourcing this great work!

