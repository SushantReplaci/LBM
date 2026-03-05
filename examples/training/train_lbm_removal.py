import datetime
import logging
import os
import random
import re
import shutil
from typing import List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import braceexpand
import fire
import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusionXLPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchvision.transforms import InterpolationMode
from pydantic.dataclasses import dataclass

from lbm.data.datasets import DataModule, DataModuleConfig
from lbm.data.filters import KeyFilter, KeyFilterConfig
from lbm.data.mappers import (
    BaseMapper,
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
    ResolutionBucketMapper,
    ResolutionBucketMapperConfig,
    ResolutionResizeMapper,
    ResolutionResizeMapperConfig,
)
from lbm.data.mappers.mappers_config import BaseMapperConfig
from lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)   
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger
from lbm.trainer.utils import StateDictAdapter


@dataclass
class MaskingMapperConfig(BaseMapperConfig):
    image_key: str = "image"
    mask_key: str = "mask"
    output_key: str = "masked_image"
    
    # Augmentation parameters
    dilation_prob: float = 0.2
    dilation_limit: Tuple[int, int] = (3, 15)
    erosion_prob: float = 0.2
    erosion_limit: Tuple[int, int] = (3, 10)
    dropout_prob: float = 0.3
    dropout_hole_size: int = 32
    dropout_ratio: float = 0.2


class MaskingMapper(BaseMapper):
    def __init__(self, config: MaskingMapperConfig):
        super().__init__(config)
        self.image_key = config.image_key
        self.mask_key = config.mask_key
        self.output_key = config.output_key
        self.config = config

    def __call__(self, batch: dict, *args, **kwargs) -> dict:
        image = batch[self.image_key] # (C, H, W) Tensor
        mask = batch[self.mask_key]   # (1, H, W) Tensor

        # Augment mask if training
        # We assume image is normalized to [0, 1] and mask is [0, 1]
        mask_np = mask.squeeze().cpu().numpy() # Try squeezing
        if mask_np.ndim == 3: # If still (C, H, W) where C > 1
             mask_np = mask_np[0]
        elif mask_np.ndim == 4: # If (B, C, H, W)
             mask_np = mask_np[0, 0]
        
        # 1. Random Dilation
        if random.random() < self.config.dilation_prob:
            kernel_size = random.randint(*self.config.dilation_limit)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_np = cv2.dilate(mask_np.astype(np.uint8), kernel, iterations=1).astype(np.float32)

        # 2. Random Erosion
        if random.random() < self.config.erosion_prob:
            kernel_size = random.randint(*self.config.erosion_limit)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_np = cv2.erode(mask_np.astype(np.uint8), kernel, iterations=1).astype(np.float32)

        # 3. Coarse Dropout (inside mask)
        if random.random() < self.config.dropout_prob:
            # We want to drop out parts of the mask, making them 0
            # This forces the model to recover original pixels too
            h, w = mask_np.shape
            aug = A.CoarseDropout(
                num_holes_range=(1, 8), 
                hole_height_range=(1, self.config.dropout_hole_size), 
                hole_width_range=(1, self.config.dropout_hole_size),
                p=1.0
            )
            # Apply only to mask where it is 1
            mask_aug = aug(image=mask_np)["image"]
            mask_np = mask_aug

        # Convert back to tensor
        mask = torch.from_numpy(mask_np).to(image.device).unsqueeze(0).float()
        
        # Injection
        noise = torch.rand_like(image)
        batch[self.output_key] = image * (1.0 - mask) + noise * mask
        
        # Update mask in batch if needed (some models might use the augmented mask)
        # batch[self.mask_key] = mask 
        
        return batch


def get_model(
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    timestep_sampling: str = "custom_timesteps",
    selected_timesteps: Optional[List[float]] = [0.0, 250.0, 500.0, 750.0],
    prob: Optional[List[float]] = [0.25, 0.25, 0.25, 0.25],
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    source_key: str = "masked_image",
    target_key: str = "target",
    mask_key: str = "mask",
    bridge_noise_sigma: float = 0.05,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 10.0,
):

    conditioners = []

    # Load pretrained model as base
    pipe = StableDiffusionXLPipeline.from_pretrained(
        backbone_signature,
        torch_dtype=torch.bfloat16,
    )

    ### MMMDiT ###
    # Get Architecture
    denoiser = DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,  # Add downsampled_image
        out_channels=vae_num_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280],
        transformer_layers_per_block=[1, 2, 10],
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=[5, 10, 20],
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=True,
        class_embed_type=None,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        num_class_embeds=None,
        upcast_attention=None,
        resnet_time_scale_shift="default",
        resnet_skip_time_act=False,
        resnet_out_scale_factor=1.0,
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        attention_type="default",
        class_embeddings_concat=False,
        mid_block_only_cross_attention=None,
        cross_attention_norm=None,
        addition_embed_type_num_heads=64,
    ).to(torch.bfloat16)

    state_dict = pipe.unet.state_dict()

    del state_dict["add_embedding.linear_1.weight"]
    del state_dict["add_embedding.linear_1.bias"]
    del state_dict["add_embedding.linear_2.weight"]
    del state_dict["add_embedding.linear_2.bias"]

    # Adapt the shapes
    state_dict_adapter = StateDictAdapter()
    state_dict = state_dict_adapter(
        model_state_dict=denoiser.state_dict(),
        checkpoint_state_dict=state_dict,
        regex_keys=[
            r"class_embedding.linear_\d+.(weight|bias)",
            r"conv_in.weight",
            r"(down_blocks|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
            r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
        ],
        strategy="zeros",
    )

    denoiser.load_state_dict(state_dict, strict=True)

    del pipe


    #concat edge maps and original image here later

    if conditioning_images_keys != [] or conditioning_masks_keys != []:

        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)

    # Wrap conditioners and set to device
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    )

    ## VAE ##
    # Get VAE model
    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()
    vae.to(torch.bfloat16)

    # LBM Config
    config = LBMConfig(
        ucg_keys=None,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    training_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )
    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )

    # LBM Model
    model = LBMModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=training_noise_scheduler,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch.bfloat16)

    return model


def get_filter_mappers(use_bucketing: bool = False):
    mappers = [
        KeyRenameMapper(
            KeyRenameMapperConfig(
                key_map={
                    "original.jpg": "image",
                    "target.png": "target",
                    "mask.png": "mask",
                }
            )
        )
    ]

    if use_bucketing:
        mappers.append(
            ResolutionBucketMapper(
                ResolutionBucketMapperConfig(key="image")
            )
        )
        mappers.append(
            ResolutionResizeMapper(
                ResolutionResizeMapperConfig(key="image", output_key="image")
            )
        )
        mappers.append(
            ResolutionResizeMapper(
                ResolutionResizeMapperConfig(key="target", output_key="target")
            )
        )
        mappers.append(
            ResolutionResizeMapper(
                ResolutionResizeMapperConfig(key="mask", output_key="mask", interpolation="nearest")
            )
        )
        # Convert to Tensors
        for key in ["image", "target", "mask"]:
            mappers.append(
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key=key,
                        transforms=["ToTensor"],
                        transforms_kwargs=[{}],
                    )
                )
            )
    else:
        # Fixed 1024x1024
        for key, interp in [("image", InterpolationMode.BILINEAR), ("target", InterpolationMode.BILINEAR), ("mask", InterpolationMode.NEAREST_EXACT)]:
            mappers.append(
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key=key,
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {"size": (1024, 1024), "interpolation": interp},
                        ],
                    )
                )
            )

    mappers.append(
        MaskingMapper(
            MaskingMapperConfig(
                image_key="image", mask_key="mask", output_key="masked_image"
            )
        )
    )
    mappers.append(RescaleMapper(RescaleMapperConfig(key="target")))
    mappers.append(RescaleMapper(RescaleMapperConfig(key="masked_image")))

    filters_mappers = [
        KeyFilter(KeyFilterConfig(keys=["original.jpg", "target.png", "mask.png"])),
        MapperWrapper(mappers),
    ]

    return filters_mappers


def get_data_module(
    train_shards: Union[List[str], List[List[str]]],
    validation_shards: Union[List[str], List[List[str]]],
    batch_size: int,
    use_bucketing: bool = False,
    mixing_probabilities: Optional[List[float]] = None,
):

    # TRAIN
    train_filters_mappers = get_filter_mappers(use_bucketing=use_bucketing)

    # unbrace urls
    if isinstance(train_shards[0], list):
        # Multiple shard lists for mixing
        train_shards_unbraced = []
        for s_list in train_shards:
            unbraced = []
            for s in s_list:
                unbraced.extend(braceexpand.braceexpand(s))
            random.shuffle(unbraced)
            train_shards_unbraced.append(unbraced)
    else:
        # Single shard list
        train_shards_unbraced = []
        for s in train_shards:
            train_shards_unbraced.extend(braceexpand.braceexpand(s))
        random.shuffle(train_shards_unbraced)

    # data config
    train_data_config = DataModuleConfig(
        shards_path_or_urls=train_shards_unbraced,
        decoder="pil",
        per_worker_batch_size=batch_size,
        num_workers=max(1, min(10, len(train_shards_unbraced) if not isinstance(train_shards_unbraced[0], list) else 10)),
        use_bucketing=use_bucketing,
        mixing_probabilities=mixing_probabilities,
        budgets=[256**2, 512**2, 768**2, 1024**2],
        probabilities=[0.1, 0.2, 0.2, 0.5],
        base_batch_sizes=[32, 16, 8, 4]
    )

    # VALIDATION (Skip bucketing for validation usually, or use fixed size)
    validation_filters_mappers = get_filter_mappers(use_bucketing=False)
    
    validation_shards_unbraced = []
    for s in validation_shards:
        validation_shards_unbraced.extend(braceexpand.braceexpand(s))

    validation_data_config = DataModuleConfig(
        shards_path_or_urls=validation_shards_unbraced,
        decoder="pil",
        per_worker_batch_size=batch_size,
        num_workers=1,
    )

    # data module
    data_module = DataModule(
        train_config=train_data_config,
        train_filters_mappers=train_filters_mappers,
        eval_config=validation_data_config,
        eval_filters_mappers=validation_filters_mappers,
    )

    return data_module


def main(
    train_shards: List[str] = ["pipe:cat path/to/train/shards"],
    validation_shards: List[str] = ["pipe:cat path/to/validation/shards"],
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0", #change to diffusers/stable-diffusion-xl-1.0-inpainting-0.1
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    source_key: str = "masked_image",
    target_key: str = "target",
    mask_key: str = "mask",
    wandb_project: str = "lbm-removal",
    batch_size: int = 8,
    num_steps: List[int] = [1, 2, 4],
    learning_rate: float = 3e-5,
    learning_rate_scheduler: str = None,
    learning_rate_scheduler_kwargs: dict = {},
    optimizer: str = "AdamW",
    optimizer_kwargs: dict = {},
    timestep_sampling: str = "custom_timesteps",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 10.0,
    selected_timesteps: List[float] = [0.0, 250.0, 500.0, 750.0],
    prob: List[float] = [0.25, 0.25, 0.25, 0.25],
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    save_ckpt_path: str = "./checkpoints",
    log_interval: int = 100,
    resume_from_checkpoint: bool = True,
    max_epochs: int = 100,
    max_steps: int = 20000,
    bridge_noise_sigma: float = 0.05,
    save_interval: int = 1000,
    path_config: str = None,
    config_yaml: dict = {},
    use_bucketing: bool = False,
    mixing_probabilities: Optional[List[float]] = None,
):
    model = get_model(
        backbone_signature=backbone_signature,
        vae_num_channels=vae_num_channels,
        unet_input_channels=unet_input_channels,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        pixel_loss_type=pixel_loss_type,
        latent_loss_type=latent_loss_type,
        latent_loss_weight=latent_loss_weight,
        pixel_loss_weight=pixel_loss_weight,
        selected_timesteps=selected_timesteps,
        prob=prob,
        conditioning_images_keys=conditioning_images_keys,
        conditioning_masks_keys=conditioning_masks_keys,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    data_module = get_data_module(
        train_shards=train_shards,
        validation_shards=validation_shards,
        batch_size=batch_size,
        use_bucketing=use_bucketing,
        mixing_probabilities=mixing_probabilities,
    )

    train_parameters = ["denoiser.*"]

    # Training Config
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["masked_image", "target", "mask"],
        trainable_params=train_parameters,
        optimizer_name=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_samples_model_kwargs={
            "input_shape": None,
            "num_steps": num_steps,
        },
    )
    if (
        os.path.exists(save_ckpt_path)
        and resume_from_checkpoint
        and "last.ckpt" in os.listdir(save_ckpt_path)
    ):
        start_ckpt = f"{save_ckpt_path}/last.ckpt"
        print(f"Resuming from checkpoint: {start_ckpt}")

    else:
        start_ckpt = None

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    pipeline.save_hyperparameters(
        {
            f"embedder_{i}": embedder.config.to_dict()
            for i, embedder in enumerate(model.conditioner.conditioners)
        }
    )

    pipeline.save_hyperparameters(
        {
            "denoiser": model.denoiser.config,
            "vae": model.vae.config.to_dict(),
            "config_yaml": config_yaml,
            "training": training_config.to_dict(),
            "training_noise_scheduler": model.training_noise_scheduler.config,
            "sampling_noise_scheduler": model.sampling_noise_scheduler.config,
        }
    )

    training_signature = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "-LBM-Removal"
        + f"{os.environ.get('SLURM_JOB_ID', 'local')}"
        + f"_{os.environ.get('SLURM_ARRAY_TASK_ID', 0)}"
    )
    dir_path = f"{save_ckpt_path}/logs/{training_signature}"
    if os.environ.get("SLURM_PROCID", "0") == "0":
        os.makedirs(dir_path, exist_ok=True)
        if path_config is not None:
            shutil.copy(path_config, f"{save_ckpt_path}/config.yaml")
    run_name = training_signature

    # Ignore parameters unused during training
    ignore_states = []
    for name, param in pipeline.model.named_parameters():
        ignore = True
        for regex in ["denoiser."]:
            pattern = re.compile(regex)
            if re.match(pattern, name):
                ignore = False
        if ignore:
            ignore_states.append(param)

    # FSDP Strategy
    strategy = FSDPStrategy(
        auto_wrap_policy=ModuleWrapPolicy(
            [
                UNet2DConditionModel,
                BasicTransformerBlock,
                ResnetBlock2D,
                torch.nn.Conv2d,
            ]
        ),
        activation_checkpointing_policy=ModuleWrapPolicy(
            [
                BasicTransformerBlock,
                ResnetBlock2D,
            ]
        ),
        sharding_strategy="SHARD_GRAD_OP",
        ignored_states=ignore_states,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=int(os.environ.get("SLURM_NPROCS", 1)) // int(os.environ.get("SLURM_NNODES", 1)),
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        strategy=strategy,
        default_root_dir="logs",
        logger=loggers.WandbLogger(
            project=wandb_project, offline=False, name=run_name, save_dir=save_ckpt_path
        ),
        callbacks=[
            WandbSampleLogger(log_batch_freq=log_interval),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=save_ckpt_path,
                every_n_train_steps=save_interval,
                save_last=True,
            ),
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        limit_val_batches=2,
        val_check_interval=1000,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )

    trainer.fit(pipeline, data_module, ckpt_path=start_ckpt)


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    logging.info(
        f"Running main with config: {yaml.dump(config, default_flow_style=False)}"
    )
    main(**config, config_yaml=config, path_config=path_config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
