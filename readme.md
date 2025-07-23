# TTS-VAR: A Test-Time Scaling Framework for Visual Auto-Regressive Generation

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)  ![GitHub stars](https://img.shields.io/github/stars/ali-vilab/TTS-VAR.svg?style=social)

## Quick Start

1. Prepare the environment
   ```bash
   cd TTS-VAR
   conda create -n tts-var python==3.10.12
   conda activate tts-var
   ```
2. Clone submodule [Infinity](https://github.com/FoundationVision/Infinity)
   ```bash
   git submodule init
   git submodule update
   ```
3. Install dependencies
   ```bash
   cd Infinity
   pip install torch==2.5.1
   pip install -r requirements.txt

   cd ..
   pip install -r requirements.txt
   ```
4. Download the pretrained models and put it in `./pretrained_models/`
   ```bash
   git-lfs install
   bash download_models.bash
   ```
5. Inference with TTS-VAR
   ```bash
   bash run.bash
   ```

## Inference Args

The inference arguments are list in `run.bash`, with specific explanation in `tts-var/main.py`. Here we show arguments for TTS-VAR. For Infinity's arguments, please refer to [Infinity](https://github.com/FoundationVision/Infinity).

### Process Arguements

| Argument.          | Description                                                   | Default                                     | Explanation              |
| ------------------ | ------------------------------------------------------------- | ------------------------------------------- | ------------------------ |
| `--reward_type`  | Type of reward model to use for reward function.              | `ir`                                      | ImageReward              |
| `--cal_type`     | Calculation type for reward-based resampling.                 | `value`                                   |                          |
| `--resample_sis` | List of step indices for resampling during inference.         | `[6,9]`                                   |                          |
| `--extract_type` | Feature extraction type for reward guidance.                  | `pca`                                     | PCA from DINOv2 features |
| `--extract_sis`  | List of step indices for feature clustering during inference. | `[2,5]`                                   |                          |
| `--bs_sis`       | List of batch sizes for different scales of generation.       | `[8, 8, 6, 6, 6, 4, 2, 2, 2, 1, 1, 1, 1]` | For 13 scales            |
| `--lam`          | Lambda value for potential weights in resampling.             | `10.0`                                    |                          |

### Path Arguements

| Argument.                      | Description                                                  | Default                                              |
| ------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------- |
| `--aesthetic_predictor_path` | Path to aesthetic predictor model when using `aes` reward. | `./pretrained_models/aesthetic_predictor_v2_5.pth` |
| `--siglip_encoder_path`      | Path to SigLIP encoder model when using `aes` reward.      | `./pretrained_models/siglip-so400m-patch14-384`    |
| `--image_reward_model_path`  | Path to ImageReward model when using `ir` reward.          | `./pretrained_models/ImageReward.pt`               |
| `--dinov2_hub_path`          | Github or locol repo of DINOv2                               | `facebookresearch/dinov2`                          |

## Acknowledgement

This code is built on top of [Infinity](https://github.com/FoundationVision/Infinity). We thank the authors for their great work.

## Citation

If you find this code useful for your research, please cite our paper:

```bash
TODO
```
