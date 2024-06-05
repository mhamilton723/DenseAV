# Separating the "Chirp" from the "Chat": Self-supervised Visual Grounding of Sound and Language
###  CVPR 2024


[![Website](https://img.shields.io/badge/DenseAV-%F0%9F%8C%90Website-purple?style=flat)](https://aka.ms/denseav) [![arXiv](https://img.shields.io/badge/arXiv-2403.10516-b31b1b.svg)](https://arxiv.org/abs/2403.10516) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhamilton723/DenseAV/blob/main/demo.ipynb)
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FeatUp-orange)](https://huggingface.co/spaces/mhamilton723/DenseAV) 
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper%20Page-orange)](https://huggingface.co/papers/2403.10516)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/featup-a-model-agnostic-framework-for/feature-upsampling-on-imagenet)](https://paperswithcode.com/sota/feature-upsampling-on-imagenet?p=featup-a-model-agnostic-framework-for)


[Mark Hamilton](https://mhamilton.net/),
[Andrew Zisserman](https://www.robots.ox.ac.uk/~az/),
[John R. Hershey](https://research.google/people/john-hershey/),
[William T. Freeman](https://billf.mit.edu/about/bio)

![DenseAV Overview Graphic](https://mhamilton.net/images/website_hero_small-p-1080.jpg)

*TL;DR*:Our model, DenseAV, learns the meaning of words and the location of sounds (visual grounding) without supervision or text.

https://github.com/mhamilton723/DenseAV/assets/6456637/ba908ab5-9618-42f9-8d7a-30ecb009091f


## Contents
<!--ts-->
   * [Install](#install)
   * [Model Zoo](#model-zoo)
   * [Evaluate Models](#evaluate-models)
   * [Train a Model](#train-model)
   * [Local Gradio Demo](#local-gradio-demo)
   * [Coming Soon](coming-soon)
   * [Citation](#citation)
   * [Contact](#contact)
<!--te-->

## Install
To use DenseAV locally clone the repository:
```shell script
git clone https://github.com/mhamilton723/FeatUp.git
cd FeatUp
```


## Getting Datasets

### Speech and Sound Prompted ADE20K

### Places Audio

### Audioset


## Model Zoo

To see examples of pretrained model usage please see our [Collab notebook](https://colab.research.google.com/github/mhamilton723/DenseAV/blob/main/demo.ipynb). We currently supply the following pretrained models:

| Model Name | Checkpoint                                                                                                                       | Checkpoint (No LayerNorm)                                                                                                                  | Torch Hub Repository | Torch Hub Name |
|------------|----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|----------------------|----------------|
| DINO       | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/dino16_jbu_stack_cocostuff.ckpt) | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/no_norm/dino16_jbu_stack_cocostuff.ckpt)   | mhamilton723/FeatUp  | dino16         |
| DINO v2    | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/dinov2_jbu_stack_cocostuff.ckpt) | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/no_norm/dinov2_jbu_stack_cocostuff.ckpt)   | mhamilton723/FeatUp  | dinov2         |
| CLIP       | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/clip_jbu_stack_cocostuff.ckpt)   | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/no_norm/clip_jbu_stack_cocostuff.ckpt)     | mhamilton723/FeatUp  | clip           |
| MaskCLIP   | n/a                                                                                                                              | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/no_norm/maskclip_jbu_stack_cocostuff.ckpt) | mhamilton723/FeatUp  | maskclip       |
| ViT        | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/vit_jbu_stack_cocostuff.ckpt)      | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/no_norm/vit_jbu_stack_cocostuff.ckpt)      | mhamilton723/FeatUp  | vit            |
| ResNet50   | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/resnet50_jbu_stack_cocostuff.ckpt) | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/no_norm/resnet50_jbu_stack_cocostuff.ckpt) | mhamilton723/FeatUp  | resnet50       |

For example, to load the FeatUp JBU upsampler for the DINO backbone without an additional LayerNorm on the spatial features:

```python
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=False)
```

To load upsamplers trained on backbones with additional LayerNorm operations which makes training and transfer learning a bit more stable:

```python
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dino16')
```

## Evaluate Models

To train an implicit upsampler for a given image and backbone first clone the repository and install it for 
[local development](#local-development). Then run

```python
cd featup
python train_implicit_upsampler.py
```

Parameters for this training operation can be found in the [implicit_upsampler config file](featup/configs/implicit_upsampler.yaml).


## Train a Model

## Local Gradio Demo

To run our [HuggingFace Spaces hosted FeatUp demo](https://huggingface.co/spaces/mhamilton723/FeatUp) locally first install FeatUp for local development. Then  run:

```shell
python gradio_app.py
```

Wait a few seconds for the demo to spin up, then navigate to [http://localhost:7860/](http://localhost:7860/) to view the demo.


## Coming Soon:

- Training your own DenseAV Model


## Citation

```
@article{hamilton2024separating,
    title={Separating the "Chirp" from the "Chat": Self-supervised Visual Grounding of Sound and Language},
    author={Hamilton, Mark and Zisserman, Andrew and Hershey, John and Freeman, William},
    journal={TODO},
    year={2024}
    }
```

## Contact

For feedback, questions, or press inquiries please contact [Mark Hamilton](mailto:markth@mit.edu)
