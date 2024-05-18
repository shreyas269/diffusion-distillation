# Enhancing On-Device Personalization of Text-to-Image Diffusion Models using Task-Specific Distillation

## Introduction

Recent advancements in large-scale text-to-image models have revolutionized image creation, enabling the generation of high-quality, diverse images from text descriptions. These models excel in semantic understanding due to extensive datasets pairing images with captions. However, they often struggle to replicate specific subjects from reference collections and create new variations in various scenarios. This project aims to address this limitation by leveraging memory-optimized knowledge distillation techniques to improve the on-device personalization of generative AI models, particularly text-to-image diffusion models.

## Features

- **Efficient Distillation**: Utilizes memory-optimized knowledge distillation to create compact models suitable for on-device adaptation.
- **Personalization**: Allows fine-tuning of pre-trained models on client devices using proprietary data without compromising data privacy.
- **Performance**: Maintains high fidelity and performance metrics while significantly reducing the computational footprint.

## Installation

### Prerequisites

Before running the script, make sure you install the Huggingface Diffusers library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Navigate to the example folder with the training script and install the required dependencies:

```bash
cd examples/custom_diffusion
pip install -r requirements.txt
pip install clip-retrieval
```

This is the starter code to train the custom diffusion model:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"
export INSTANCE_DIR="./data/cat"

accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./real_reg/samples_cat/ \
  --with_prior_preservation \
  --real_prior \
  --prior_loss_weight=1.0 \
  --class_prompt="cat" \
  --num_class_images=200 \
  --instance_prompt="photo of a <new1> cat"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --scale_lr \
  --hflip  \
  --modifier_token "<new1>" \
  --validation_prompt="<new1> cat sitting in a bucket" \
  --report_to="wandb" \
  --push_to_hub
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
  --output_dir="<Enter your dir name>"
```

### Training Settings

- Learning Rate: `1e-5`
- Scheduler: `cosine`
- Batch Size: `32`
- Output Weight: `0.5`
- Feature Weight: `0.5`


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
