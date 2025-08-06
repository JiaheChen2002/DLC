# Mitigating Hallucination of Large Vision-Language Models via Dynamic Logits Calibration

[![arXiv](https://img.shields.io/badge/arXiv-2506.21509-red)](https://arxiv.org/abs/2506.21509)
[![Code](https://img.shields.io/badge/Code-Available-green)](https://github.com/JiaheChen2002/DLC)

> **Official PyTorch implementation of "Mitigating Hallucination of Large Vision-Language Models via Dynamic Logits Calibration"**


## 🎉 News

- **[2025.06]** 🔥 Our paper is now available on arXiv!
- **[2024.08]** ⌨ We release the relevant code！

## 🎯 Overview

## ⚙️ Getting Started

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate DLC
```

### 2. Prepare Datasets
Please download the following datasets and place them in your preferred data directory.
- [COCO](https://cocodataset.org/#download)

- [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)

- [LLaVA-Bench-in-the-Wild](https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild)


### 3. Download Models
You will need to download several pretrained models and update their paths in the corresponding configuration files.

- Download [LLaVA 7B v1.5 model](https://huggingface.co/liuhaotian/llava-v1.5-7b）) and specify it at `eval_configs/llava-1.5_eval.yaml`.

- Download [Vicuna 7B v1.1 model](https://github.com/lm-sys/FastChat) and specify it at `minigpt4/configs/models/blip2_instruct_vicuna7b.yaml`. 

- Download [Vicuna 7B v0 model](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) and specify it at of `minigpt4/configs/models/minigpt4_vicuna0.yaml`. 

- Download [MiniGPT-4 7B pretrained weights](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view) and specify it at `eval_configs/minigpt4_eval.yaml`. 

### 4.Download CLIP Models
Download the CLIP model and specify its path for the `--clip-model-path` parameter within the script's arguments (args).

- [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)
- [fg-clip-large](https://huggingface.co/qihoo360/fg-clip-large)

## 🕹️ Usage
### CHAIR
After setting up the environment, datasets, and models, you can run `DLC_generate_caption.py` to perform inference with DLC.

The core implementation of our Dynamic Logits Calibration method can be found in DLC_logits_processor.py.

Here is an example command to run DLC decoding:
```bash
python DLC_generate_caption.py \
    --sample \
    --use_dlc \
    --data-path <your data path> \
    --output-dir <your output path> \
    --clip-model-path <clip model path> \
    --tokenizer-path <model path>
```

### POPE
Running `pope_eval.py` can evaluate the DLC on the pope benchmark

Here is an example command to run DLC decoding:
```bash
python pope_eval.py --sample --use_dlc --pope-type coco_popular --data-path <your data path>
```

### GPT4o-assisted
You can run the `gpt4o_eval.py` file to compare the generated content between the two methods using gpt4o. You need to configure your API key in this file.

Here is an example command to run DLC decoding:
```bash
python gpt4o_eval.py --method1_file <your method1 path> --method2_file <your method2 path> --data_path <your data path> --method1_name <your method1 name> --method2_name <your method2 name> 
```

### SHR

The method of running SHR is similar, but the price of GPT-4 is very high, so it is not very recommended to continue using it.

The code is in `shr_eval.py`. For more details, please refer to [SHR](https://github.com/opendatalab/HA-DPO/tree/main)


## 📑 Citation
If you find our work useful for your research, please consider starring our repository and citing our paper:
```
@article{chen2025mitigating,
  title={Mitigating Hallucination of Large Vision-Language Models via Dynamic Logits Calibration},
  author={Chen, Jiahe and He, Jiaying and Shao, Qian and Chen, Qiyuan and Ying, Jiahe and Xu, Hongxia and Chen, Jintai and Zheng, Jianwei and Wu, Jian},
  journal={arXiv preprint arXiv:2506.21509},
  year={2025}
}
```

## 📝 Acknowledgments
Our work builds upon several incredible open-source projects. We thank the authors of these projects for their valuable contributions to the community.
- [Visual Contrastive Decoding](https://github.com/DAMO-NLP-SG/VCD): Visual Contrastive Decoding
- [OPERA](https://github.com/shikiw/OPERA?tab=readme-ov-file) Over-Trust Penalty and Retrospection-Allocation
- [SID](https://github.com/huofushuo/SID) Self-Introspective Decoding
- [ICD](https://github.com/p1k0pan/ICD) Instruction Contrastive Decoding
- [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip): Towards General-purpose Vision-Language Models with Instruction Tuning
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4): Enhancing Vision-Language Understanding with Advanced Large Language Models
- [LLaVA 1.5](https://github.com/haotian-liu/LLaVA): Visual Instruction Tuning
