"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import sys
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.parallel
import tqdm
import wandb
import yaml

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    # data_loader = _data_loader.create_data_loader(config, framework="pytorch", shuffle=True)
    data_loader = _data_loader.create_torch_behavior_data_loader(
        config,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        skip_norm_stats=False,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, extra_modules=None):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        if extra_modules is not None:
            for name, module in extra_modules.items():
                torch.save(module.state_dict(), tmp_ckpt_dir / f"{name}.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device, extra_modules=None):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        if extra_modules is not None:
            for name, module in extra_modules.items():
                path = ckpt_dir / f"{name}.pt"
                if path.exists():
                    module.load_state_dict(torch.load(path, map_location=device, weights_only=False))
                    logging.info(f"Loaded extra module {name} from {path}")
                else:
                    logging.warning(f"Extra module checkpoint not found: {path}")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def freeze_paligemma_for_joint_training(model):
    """Freeze the VLM backbone while leaving the action expert/projections trainable."""
    model_to_freeze = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    for param in model_to_freeze.paligemma_with_expert.paligemma.parameters():
        param.requires_grad_(False)

    trainable = sum(p.numel() for p in model_to_freeze.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model_to_freeze.parameters() if not p.requires_grad)
    logging.info(f"Frozen PaliGemma backbone for joint training: trainable={trainable:,}, frozen={frozen:,}")


def setup_jepa_components(config: _config.TrainConfig, device: torch.device):
    if config.jepa_loss_weight <= 0:
        return None
    if config.jepa_vjepa2_root is None:
        raise ValueError("jepa_vjepa2_root must be set when jepa_loss_weight > 0")
    if config.jepa_encoder_ckpt is None:
        raise ValueError("jepa_encoder_ckpt must be set when jepa_loss_weight > 0")
    if config.jepa_predictor_ckpt is None:
        raise ValueError("jepa_predictor_ckpt must be set when jepa_loss_weight > 0")

    vjepa2_root = os.path.abspath(config.jepa_vjepa2_root)
    if vjepa2_root not in sys.path:
        sys.path.insert(0, vjepa2_root)

    from app.vjepa_behavior.encode_latents import encode_frame, load_encoder
    from src.models.ac_predictor import vit_ac_predictor

    predictor_config_path = config.jepa_predictor_config
    if not os.path.isabs(predictor_config_path):
        predictor_config_path = os.path.join(vjepa2_root, predictor_config_path)
    with open(predictor_config_path) as f:
        jepa_cfg = yaml.safe_load(f)

    pred_cfg = jepa_cfg["model"]
    predictor = vit_ac_predictor(
        img_size=jepa_cfg["data"]["crop_size"],
        patch_size=jepa_cfg["data"]["patch_size"],
        num_frames=2,
        tubelet_size=jepa_cfg["data"]["tubelet_size"],
        embed_dim=1024,
        predictor_embed_dim=pred_cfg["pred_embed_dim"],
        depth=pred_cfg["pred_depth"],
        num_heads=pred_cfg["pred_num_heads"],
        is_frame_causal=True,
        use_rope=pred_cfg.get("use_rope", True),
        use_activation_checkpointing=pred_cfg.get("use_activation_checkpointing", False),
        action_embed_dim=32 * config.jepa_action_dim,
        state_embed_dim=256,
        use_extrinsics=False,
    ).to(device)

    ckpt = torch.load(config.jepa_predictor_ckpt, map_location="cpu", weights_only=False)
    predictor.load_state_dict(ckpt["predictor"] if "predictor" in ckpt else ckpt)
    predictor.train()

    encoder = load_encoder(config.jepa_encoder_ckpt, device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    return {
        "predictor": predictor,
        "encoder": encoder,
        "encode_frame": encode_frame,
    }


def prepare_jepa_frames(images: torch.Tensor, device: torch.device, size: int = 256) -> torch.Tensor:
    images = images.to(device)
    if images.ndim == 5 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim != 4:
        raise ValueError(f"Expected JEPA images with 4 dims, got {tuple(images.shape)}")
    if images.shape[-1] in (1, 3):
        images = images.permute(0, 3, 1, 2)
    images = images.to(torch.float32)
    if images.max() > 2.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = images / 2.0 + 0.5
    if images.shape[-2:] != (size, size):
        images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=images.dtype, device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=images.dtype, device=device)[None, :, None, None]
    return (images - mean) / std


def unnormalize_actions_for_jepa(actions: torch.Tensor, data_config: _config.DataConfig, action_dim: int) -> torch.Tensor:
    actions = actions[..., :action_dim].to(torch.float32)
    stats = None if data_config.norm_stats is None else data_config.norm_stats.get("actions")
    if stats is None:
        return actions

    if data_config.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
        q01 = torch.as_tensor(stats.q01[..., :action_dim], dtype=actions.dtype, device=actions.device)
        q99 = torch.as_tensor(stats.q99[..., :action_dim], dtype=actions.dtype, device=actions.device)
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    mean = torch.as_tensor(stats.mean[..., :action_dim], dtype=actions.dtype, device=actions.device)
    std = torch.as_tensor(stats.std[..., :action_dim], dtype=actions.dtype, device=actions.device)
    return actions * (std + 1e-6) + mean


def compute_jepa_loss(jepa_components, jepa_batch, action_chunk, device):
    encoder = jepa_components["encoder"]
    predictor = jepa_components["predictor"]
    encode_frame = jepa_components["encode_frame"]

    frame_t = prepare_jepa_frames(jepa_batch["current_image"], device)
    frame_tH = prepare_jepa_frames(jepa_batch["future_image"], device)
    state = jepa_batch["state"].to(device, dtype=torch.float32)
    action_chunk = action_chunk.to(device, dtype=torch.float32)

    with torch.no_grad():
        z_t = F.layer_norm(encode_frame(encoder, frame_t, device), [1024])
        z_tH = F.layer_norm(encode_frame(encoder, frame_tH, device), [1024])

    action_flat = action_chunk.reshape(action_chunk.shape[0], -1).unsqueeze(1)
    state_in = state.unsqueeze(1)
    action_flat = F.layer_norm(action_flat, action_flat.shape[-1:])
    state_in = F.layer_norm(state_in, state_in.shape[-1:])
    z_pred = predictor(z_t, action_flat, state_in)
    return torch.mean(torch.abs(z_pred - z_tH))


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # # Log sample images to wandb on first batch
    # if is_main and config.wandb_enabled and not resuming:
    #     # Create a separate data loader for sample batch to avoid consuming the main loader
    #     sample_data_loader = _data_loader.create_data_loader(config, framework="pytorch", shuffle=False)
    #     sample_batch = next(iter(sample_data_loader))
    #     # Convert observation and actions to torch tensors
    #     observation, actions = sample_batch
    #     sample_batch = observation.to_dict()
    #     sample_batch["actions"] = actions

    #     # Create sample images for wandb
    #     images_to_log = []
    #     # Get batch size from the first image tensor
    #     batch_size = next(iter(sample_batch["image"].values())).shape[0]
    #     for i in range(min(5, batch_size)):
    #         # Concatenate all camera views horizontally for this batch item
    #         # Convert from NCHW to NHWC format for wandb
    #         img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
    #         img_concatenated = img_concatenated.cpu().numpy()
    #         images_to_log.append(wandb.Image(img_concatenated))

    #     wandb.log({"camera_views": images_to_log}, step=0)

    #     # Clear sample batch from memory aggressively
    #     del sample_batch, observation, actions, images_to_log, img_concatenated
    #     del sample_data_loader  # Also delete the sample data loader
    #     gc.collect()
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #     logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    if config.pytorch_freeze_paligemma or config.jepa_loss_weight > 0:
        freeze_paligemma_for_joint_training(model)

    jepa_components = setup_jepa_components(config, device)
    extra_modules = None
    if jepa_components is not None:
        extra_modules = {"jepa_predictor": jepa_components["predictor"]}
        logging.info(
            f"Enabled V-JEPA joint loss: lambda={config.jepa_loss_weight}, anneal_steps={config.jepa_action_anneal_steps}"
        )

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    optim_params = [p for p in get_model_parameters(model) if p.requires_grad]
    if jepa_components is not None:
        optim_params.extend(p for p in jepa_components["predictor"].parameters() if p.requires_grad)

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        optim_params,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device, extra_modules=extra_modules)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for batch in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            if len(batch) == 2:
                observation, actions = batch
                jepa_batch = None
            elif len(batch) == 3:
                observation, actions, jepa_batch = batch
            else:
                raise ValueError(f"Unexpected batch structure with {len(batch)} entries.")

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            use_jepa = jepa_components is not None and jepa_batch is not None
            model_out = model(observation, actions, return_action_pred=use_jepa)
            if use_jepa:
                losses, action_pred_norm = model_out
            else:
                losses = model_out
                action_pred_norm = None

            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            flow_loss = losses.mean()
            jepa_loss = None
            if use_jepa:
                pred_weight = min(1.0, global_step / max(1, config.jepa_action_anneal_steps))
                action_for_jepa_norm = (1.0 - pred_weight) * actions.detach() + pred_weight * action_pred_norm
                action_for_jepa = unnormalize_actions_for_jepa(action_for_jepa_norm, data_config, config.jepa_action_dim)
                jepa_loss = compute_jepa_loss(jepa_components, jepa_batch, action_for_jepa, device)
                loss = flow_loss + config.jepa_loss_weight * jepa_loss
            else:
                loss = flow_loss

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(optim_params, max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "flow_loss": flow_loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )
                if jepa_loss is not None:
                    infos[-1]["jepa_loss"] = jepa_loss.item()

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_flow_loss = sum(info["flow_loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                avg_jepa_loss = None
                if any("jepa_loss" in info for info in infos):
                    vals = [info["jepa_loss"] for info in infos if "jepa_loss" in info]
                    avg_jepa_loss = sum(vals) / len(vals)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                msg = f"step={global_step} loss={avg_loss:.4f} flow={avg_flow_loss:.4f}"
                if avg_jepa_loss is not None:
                    msg += f" jepa={avg_jepa_loss:.4f}"
                msg += f" lr={avg_lr:.2e}"
                if avg_grad_norm is not None:
                    msg += f" grad_norm={avg_grad_norm:.2f}"
                msg += f" time={elapsed:.1f}s"
                logging.info(msg)

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "flow_loss": avg_flow_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_jepa_loss is not None:
                        log_payload["jepa_loss"] = avg_jepa_loss
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config, extra_modules=extra_modules)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
