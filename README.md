# Diffusion-Transformer
In this Project, a fine tuned Diffusion Transformer (DiT) is used for image generation using PyTorch Distributed Data Parallel (DDP). 
In this project we have used a pre-trained DiT (Diffusion Transformer) for image generation and fine-tuned the model on a dataset (e.g., CelebA-HQ) using a diffusion loss function.
The specific model used is specified in --model (e.g., "DiT-XL/2").
The DiT model is not directly pretrained in this script, but it loads a predefined architecture from DiT_models.
It uses AutoencoderKL from stabilityai/sd-vae-ft-{args.vae} to encode images into the latent space before training the diffusion model.
The script
The fine-tuning method involves:
Encoding images into latent space using a pretrained VAE (AutoencoderKL).
Applying noise to the latent representations at different timesteps.
Training DiT to predict the original image latents given the noisy latents (denoising process).
Updating parameters using AdamW optimizer.
Exponential Moving Average (EMA) is applied to stabilize training.
Saving checkpoints periodically.
