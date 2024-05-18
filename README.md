# Enhancing On-Device Personalization of Text-to-Image Diffusion Models using Task-Specific Distillation

## Introduction

Recent advancements in large-scale text-to-image models have revolutionized image creation, enabling the generation of high-quality, diverse images from text descriptions. These models excel in semantic understanding due to extensive datasets pairing images with captions. However, they often struggle to replicate specific subjects from reference collections and create new variations in various scenarios. This project aims to address this limitation by leveraging memory-optimized knowledge distillation techniques to improve the on-device personalization of generative AI models, particularly text-to-image diffusion models.

## Features

- **Efficient Distillation**: Utilizes memory-optimized knowledge distillation to create compact models suitable for on-device adaptation.
- **Personalization**: Allows fine-tuning of pre-trained models on client devices using proprietary data without compromising data privacy.
- **Performance**: Maintains high fidelity and performance metrics while significantly reducing the computational footprint.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
pip install -r requirements.txt
```

Ensure you have the Huggingface Diffusers library and Accelerate library installed:

```bash
pip install diffusers accelerate
```

## Usage

### Download Data

First, download the data required for training using `data.py`:

```bash
python data.py
```

### Distillation Training

To train the U-net using the methods described in the paper, run the distillation training script with the following command:

```bash
export MODEL_NAME="SG161222/Realistic_Vision_V4.0"
export DATASET_NAME="fantasyfish/laion-art"

accelerate launch --mixed_precision="fp16" distill_training.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --distill_level="sd_small" \
  --prepare_unet="True" \
  --output_weight=0.5 \
  --feature_weight=0.5 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=<"Paste your output dir name">
```

### Training Settings

- Learning Rate: `1e-5`
- Scheduler: `cosine`
- Batch Size: `32`
- Output Weight: `0.5`
- Feature Weight: `0.5`

## Contributing

We welcome contributions to enhance the project. Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch`
5. Submit a pull request.



## Acknowledgments

- The starter basic training code was sourced from the Huggingface Diffusers library.
- Various techniques were referenced from [Kumari et al., 2023] and [Shi et al., 2023] for enhancing the language-vision dictionary of these models.

## References

- Choi, J., Choi, Y., Kim, Y., Kim, J., & Yoon, S. (2023). Custom-Edit: Text-Guided Image Editing with Customized Diffusion Models. arXiv:2305.15779 [cs].
- Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A. H., Chechik, G., & Cohen-Or, D. (2022). An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. arXiv:2208.01618 [cs].
- Hessel, J., Holtzman, A., Forbes, M., Bras, R. L., & Choi, Y. (2022). CLIPScore: A Reference-free Evaluation Metric for Image Captioning. arXiv:2104.08718 [cs].
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531 [cs, stat].
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. arXiv:2006.11239 [cs, stat].
- Kim, B.-K., Song, H.-K., Castells, T., & Choi, S. (2023). BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion. arXiv:2305.15798 [cs].
- Kumari, N., Zhang, B., Zhang, R., Shechtman, E., & Zhu, J.-Y. (2023). Multi-Concept Customization of Text-to-Image Diffusion. arXiv:2212.04488 [cs].
- Kumari, N., Zhang, B., Zhang, R., Shechtman, E., & Zhu, J.-Y. (2024). Custom-edit: Text-guided image editing with customized diffusion models. Advances in Neural Information Processing Systems, 36.
- Li, Y., Wang, H., Jin, Q., Hu, J., Chemerys, P., Fu, Y., Wang, Y., Tulyakov, S., & Ren, J. (2023). SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds. arXiv:2306.00980 [cs].
- Meng, C., Rombach, R., Gao, R., Kingma, D. P., Ermon, S., Ho, J., & Salimans, T. (2023). On Distillation of Guided Diffusion Models. arXiv:2210.03142 [cs].
- Pernias, P., Rampas, D., Richter, M. L., Pal, C. J., & Aubreville, M. (2023). Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models. arXiv:2306.00637 [cs].
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. arXiv:2112.10752 [cs].
- Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., & Aberman, K. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. arXiv:2208.12242 [cs].
- Salimans, T., & Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models. arXiv:2202.00512 [cs, stat].
- Shi, J., Xiong, W., Lin, Z., & Jung, H. J. (2023). InstantBooth: Personalized Text-to-Image Generation without Test-Time Finetuning. arXiv:2304.03411 [cs].
