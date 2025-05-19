# Add these lines at the very top of train.py
import sys
import os
sys.path.append('/mnt/DATA/panam/daclip-uir/universal-image-restoration/open_clip')
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import patch_wandb  # This will patch wandb before any other imports

import argparse
import logging
import math
import os
import random
import sys
import copy

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# import open_clip

import options as option
from models import create_model

sys.path.insert(0, "../../")
import open_clip
import util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from data.util import bgr2ycbcr
import importlib.util

# Path to the module file
module_path = '/mnt/DATA/panam/daclip-uir/universal-image-restoration/utils/sde_utils.py'

# Load the module
spec = importlib.util.spec_from_file_location('sde_utils', module_path)
sde_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sde_utils)

# Now you can use IRSDE from the module
IRSDE = sde_utils.IRSDE

# Import the validation functions
from sae_validation import check_sparsity, visualize_feature_activations, analyze_sparse_features

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group

def initialize_sae(model, dataloader, device, num_batches=5):
    """Pre-initialize the SAE with real data to avoid poor initial convergence"""
    logger = logging.getLogger("base")
    logger.info("Pre-initializing sparse autoencoder with data distribution...")
    
    if not hasattr(model, 'update_geometric_median'):
        logger.warning("Model does not have update_geometric_median method, skipping pre-initialization")
        return
    
    try:
        # Update geometric median
        model.update_geometric_median(dataloader, device=device, num_batches=num_batches)
        logger.info("Geometric median updated successfully")
    except Exception as e:
        logger.error(f"Error in SAE pre-initialization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                    and "daclip" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")


    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt) 
    device = model.device

    # Create directories for visualizations
    os.makedirs('image', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    if opt['path']['daclip'] is not None:
        # Load model from pretrained path
        base_clip, preprocess = open_clip.create_model_from_pretrained(
            'daclip_ViT-B-32', 
            pretrained=opt['path']['daclip'],
        )
        
        from daclip_model import DaCLIP  # Import DaCLIP
        
        # Check if base_clip is already a DaCLIP model - if so, use it directly
        if hasattr(base_clip, 'visual_controller') or hasattr(base_clip, 'use_sparse_autoencoder'):
            clip_model = base_clip  # Already a DaCLIP model
            print("Using base_clip directly as it appears to be a DaCLIP model already")
        else:
            # Wrap regular CLIP in DaCLIP
            clip_model = DaCLIP(base_clip, use_sparse_autoencoder=True, l1_coeff=0.001, expansion_factor=8)
        
        # Initialize controller from base model (only once)
        if hasattr(clip_model, 'initial_controller'):
            clip_model.initial_controller()
        
        # Move to device if needed
        if torch.cuda.is_available():
            clip_model = clip_model.to(device)
    else:
        # Regular CLIP model
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        clip_model = clip_model.to(device)

    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Initialize geometric median for sparse autoencoder
    if hasattr(clip_model, 'update_geometric_median'):
        logger.info("Initializing geometric median for sparse autoencoder...")
        try:
            # Create a simple sampler for geometric median initialization
            # Initialize SAE with real data
            initialize_sae(clip_model, train_loader, device=device, num_batches=10)
            
            # Log SAE information
            if hasattr(clip_model, 'degra_sae'):
                logger.info(f"Degradation SAE: input_dim={clip_model.image_dim}, " +
                           f"hidden_dim={clip_model.degra_sae.n_learned_features}")
                
                if hasattr(clip_model, 'text_sae'):
                    logger.info(f"Text SAE: input_dim={clip_model.text_dim}, " + 
                               f"hidden_dim={clip_model.text_sae.n_learned_features}")
                
                logger.info(f"L1 coefficient: {clip_model.l1_coeff}")
        except Exception as e:
            logger.error(f"Failed to initialize geometric median: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    sde = IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)

    # Add new metrics for tracking SAE performance
    if hasattr(clip_model, 'use_sparse_autoencoder') and clip_model.use_sparse_autoencoder:
        logger.info("Sparse autoencoder is enabled.")
        
        # Run initial sparsity check
        try:
            sample_batch = next(iter(train_loader))
            img4clip = sample_batch["LQ_clip"].to(device)
            sparsity_metrics = check_sparsity(clip_model, img4clip)
            logger.info(f"Initial sparsity: {sparsity_metrics['sparsity']:.4f}")
            logger.info(f"Initial decoder norms - min: {sparsity_metrics['min_decoder_norm']:.6f}, " +
                       f"max: {sparsity_metrics['max_decoder_norm']:.6f}")
        except Exception as e:
            logger.warning(f"Initial sparsity check failed: {str(e)}")
    
    # Track feature utilization
    feature_activation_history = {}

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break

            LQ, GT, deg_type = train_data["LQ"], train_data["GT"], train_data["type"]
            if isinstance(deg_type, list):
                # Convert list of strings to a tensor of tokens
                deg_token = tokenizer(deg_type[0]).unsqueeze(0).to(device)
                for i in range(1, len(deg_type)):
                    token = tokenizer(deg_type[i]).unsqueeze(0).to(device)
                    deg_token = torch.cat([deg_token, token], dim=0)
            else:
                deg_token = tokenizer(deg_type).to(device)            
            img4clip = train_data["LQ_clip"].to(device)
            
            # Check gradients before forward pass (every 1000 steps)
            if current_step % 1000 == 0 and rank <= 0 and hasattr(clip_model, 'degra_sae'):
                # Enable gradient tracking temporarily
                with torch.set_grad_enabled(True):
                    # Zero gradients
                    if hasattr(clip_model, 'parameters'):
                        for param in clip_model.parameters():
                            if param.grad is not None:
                                param.grad.zero_()
                    
                    # Forward with just a small batch
                    mini_batch = img4clip[:1].clone().requires_grad_(True)
                    
                    # Use the correct method based on the model
                    if hasattr(clip_model, 'encode_image'):
                        result = clip_model.encode_image(mini_batch, control=True)
                    elif hasattr(clip_model, 'visual_controller') and hasattr(clip_model.visual_controller, 'forward'):
                        result = clip_model.visual_controller(mini_batch, control=True)
                    else:
                        logger.warning("Could not find appropriate encode_image method")
                        result = None
                    
                    if isinstance(result, tuple) and len(result) == 3:
                        _, degra_features, sae_loss = result
                        
                        # Multiply loss by a factor to match your training
                        sae_loss = sae_loss * 10.0

                        # Backward on the SAE loss
                        sae_loss.backward(retain_graph=True)
                        
                        # Print gradient information
                        grad_norm = 0.0
                        param_count = 0
                        
                        # Find the proper SAE parameters
                        sae_params = []
                        if hasattr(clip_model, 'degra_sae'):
                            sae_params = clip_model.degra_sae.parameters()
                        elif hasattr(clip_model, 'visual_controller') and hasattr(clip_model.visual_controller, 'degra_sae'):
                            sae_params = clip_model.visual_controller.degra_sae.parameters()
                        
                        # Log gradient information
                        for name, param in [(f"sae_param_{i}", p) for i, p in enumerate(sae_params)]:
                            if param.grad is not None:
                                grad_norm += param.grad.norm().item()
                                param_count += 1
                                if param_count < 5:  # Print first few gradients
                                    logger.info(f"Grad for {name}: {param.grad.norm().item():.6f}")
                        
                        if param_count > 0:
                            grad_norm /= param_count
                            logger.info(f"SAE average gradient norm: {grad_norm:.6f}")
                        else:
                            logger.info("No gradients found for SAE parameters!")
                        
                        # Feature statistics
                        min_val = degra_features.min().item()
                        max_val = degra_features.max().item()
                        near_zero = (torch.abs(degra_features) < 1e-5).float().mean().item() * 100.0
                        logger.info(f"Feature stats: min={min_val:.6f}, max={max_val:.6f}, near-zero={near_zero:.2f}%")
                    
                    # Zero gradients again
                    if hasattr(clip_model, 'parameters'):
                        for param in clip_model.parameters():
                            if param.grad is not None:
                                param.grad.zero_()
            
            # Regular forward pass (no grad)
            with torch.no_grad(), torch.amp.autocast('cuda'): # Updated autocast usage
                # Use the correct method based on the model
                if hasattr(clip_model, 'encode_image'):
                    result = clip_model.encode_image(img4clip, control=True)
                elif hasattr(clip_model, 'visual_controller') and hasattr(clip_model.visual_controller, 'forward'):
                    result = clip_model.visual_controller(img4clip, control=True)
                else:
                    logger.warning("Could not find appropriate encode_image method")
                    result = None

                if isinstance(result, tuple) and len(result) == 3:
                    # New format with sparse autoencoder
                    image_context, degra_context, sae_loss = result
                    sae_loss = sae_loss * 10.0
                    if opt["use_tb_logger"] and "debug" not in opt["name"] and rank <= 0 and current_step % 100 == 0:
                        tb_logger.add_scalar("sae_loss", sae_loss.item(), current_step)
                        # Log L1 loss component
                        if hasattr(clip_model, 'l1_coeff'):
                            l1_coeff = clip_model.l1_coeff
                            logger.info(f"SAE L1 coefficient: {l1_coeff:.6f}")
                else:
                    # Original format without sparse autoencoder
                    image_context, degra_context = result
                    sae_loss = None 
                # Convert to float as in your original code
                image_context = image_context.float()
                degra_context = degra_context.float()

            # Add sparse autoencoder validation
            if current_step % 1000 == 0 and rank <= 0:  # Every 1000 steps
                # Check and visualize sparsity
                try:
                    # Direct feature inspection for better debugging
                    if isinstance(result, tuple) and len(result) == 3:
                        # Directly compute sparsity from the features
                        near_zeros = (torch.abs(degra_context) < 1e-5).float().mean().item()
                        exact_zeros = (degra_context == 0.0).float().mean().item()
                        logger.info(f"Direct sparsity check: near-zero={near_zeros*100:.2f}%, exact-zero={exact_zeros*100:.2f}%")
                        
                        # Check non-zero elements
                        non_zeros = (torch.abs(degra_context) >= 1e-5).sum().item()
                        total_elements = degra_context.numel()
                        logger.info(f"Non-zero elements: {non_zeros}/{total_elements} ({non_zeros/total_elements*100:.2f}%)")
                    
                    # Use the validation function
                    sae_module = None
                    if hasattr(clip_model, 'degra_sae'):
                        sae_module = clip_model.degra_sae
                    elif hasattr(clip_model, 'visual_controller') and hasattr(clip_model.visual_controller, 'degra_sae'):
                        sae_module = clip_model.visual_controller.degra_sae
                    
                    if sae_module is not None:
                        sparsity_metrics = check_sparsity(clip_model, img4clip)
                        if 'features' in sparsity_metrics:
                            visualize_feature_activations(
                                sparsity_metrics['features'], 
                                save_path=f'visualizations/features_step_{current_step}.png'
                            )
                            if opt["use_tb_logger"] and "debug" not in opt["name"]:
                                tb_logger.add_scalar("sparsity", sparsity_metrics['sparsity'], current_step)
                                tb_logger.add_scalar("min_decoder_norm", sparsity_metrics['min_decoder_norm'], current_step)
                                tb_logger.add_scalar("max_decoder_norm", sparsity_metrics['max_decoder_norm'], current_step)
                            
                            # Log sparsity metric
                            logger.info(f"Step {current_step}: Feature sparsity = {sparsity_metrics['sparsity']:.4f}")
                            logger.info(f"Step {current_step}: Decoder norms - min: {sparsity_metrics['min_decoder_norm']:.6f}, " +
                                      f"max: {sparsity_metrics['max_decoder_norm']:.6f}")
                            
                            # Track feature utilization over time
                            if 'features' in sparsity_metrics:
                                active_features = (sparsity_metrics['features'].abs() > 1e-5).float().sum().item()
                                total_features = sparsity_metrics['features'].numel()
                                
                                # Log feature utilization
                                logger.info(f"Feature utilization: {active_features}/{total_features} " +
                                          f"({active_features/total_features*100:.2f}%)")
                                
                                # Store in history
                                feature_activation_history[current_step] = active_features / total_features
                                
                                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                                    tb_logger.add_scalar("feature_utilization", active_features/total_features, current_step)
                except Exception as e:
                    logger.error(f"Error in sparse autoencoder validation: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())

            timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)

            model.feed_data(states, LQ, GT, text_context=degra_context, image_context=image_context, sae_loss=sae_loss) # xt, mu, x0
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )
            
            # Apply post-backward hooks for sparse autoencoder
            if hasattr(clip_model, 'apply_post_backward_hooks'):
                clip_model.apply_post_backward_hooks()

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                torch.cuda.empty_cache()
                avg_psnr = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):

                    LQ, GT, deg_type = val_data["LQ"], val_data["GT"], val_data["type"]
                    if isinstance(deg_type, list):
                        # Convert list of strings to a tensor of tokens
                        deg_token = tokenizer(deg_type[0]).unsqueeze(0).to(device)
                        for i in range(1, len(deg_type)):
                            token = tokenizer(deg_type[i]).unsqueeze(0).to(device)
                            deg_token = torch.cat([deg_token, token], dim=0)
                    else:
                        deg_token = tokenizer(deg_type).to(device)                    
                    img4clip = val_data["LQ_clip"].to(device)
                    
                    with torch.no_grad(), torch.amp.autocast('cuda'): # Updated autocast usage
                        # Use the correct method based on the model
                        if hasattr(clip_model, 'encode_image'):
                            result = clip_model.encode_image(img4clip, control=True)
                        elif hasattr(clip_model, 'visual_controller') and hasattr(clip_model.visual_controller, 'forward'):
                            result = clip_model.visual_controller(img4clip, control=True)
                        else:
                            logger.warning("Could not find appropriate encode_image method")
                            result = None
                            
                        if isinstance(result, tuple) and len(result) == 3:
                            # New format with sparse autoencoder
                            image_context, degra_context, sae_loss = result
                            if opt["use_tb_logger"] and "debug" not in opt["name"] and rank <= 0 and current_step % 100 == 0:
                                tb_logger.add_scalar("val_sae_loss", sae_loss.item(), current_step)
                        else:
                            # Original format without sparse autoencoder
                            image_context, degra_context = result
                            sae_loss = None 
                        # Convert to float as in your original code
                        image_context = image_context.float()
                        degra_context = degra_context.float()

                    noisy_state = sde.noise_state(LQ)

                    # valid Predictor
                    model.feed_data(noisy_state, LQ, GT, text_context=degra_context, image_context=image_context, sae_loss=sae_loss)
                    model.test(sde)
                    visuals = model.get_current_visuals()

                    output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                    gt_img = util.tensor2img(GT.squeeze())  # uint8
                    lq_img = util.tensor2img(LQ.squeeze())

                    util.save_img(output, f'image/{idx}_{deg_type[0]}_SR.png')
                    util.save_img(gt_img, f'image/{idx}_{deg_type[0]}_GT.png')
                    util.save_img(lq_img, f'image/{idx}_{deg_type[0]}_LQ.png')

                    # calculate PSNR
                    avg_psnr += util.calculate_psnr(output, gt_img)
                    idx += 1

                    if idx > 99:
                        break

                avg_psnr = avg_psnr / idx

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step

                # log
                logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    )
                )
                print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    ))
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)

                # Perform validation with and without SAE (every 5000 steps)
                if current_step % 5000 == 0 and hasattr(clip_model, 'use_sparse_autoencoder'):
                    # Run another validation with SAE disabled
                    original_sae_mode = clip_model.use_sparse_autoencoder
                    clip_model.use_sparse_autoencoder = False
                    
                    avg_psnr_no_sae = 0.0
                    idx = 0
                    for _, val_data in enumerate(val_loader):
                        if idx >= 20:  # Only test on a subset
                            break
                            
                        LQ, GT, deg_type = val_data["LQ"], val_data["GT"], val_data["type"]
                        img4clip = val_data["LQ_clip"].to(device)
                        
                        with torch.no_grad():
                            # Use the correct method based on the model
                            if hasattr(clip_model, 'encode_image'):
                                if hasattr(clip_model, 'use_sparse_autoencoder') and clip_model.use_sparse_autoencoder:
                                    image_context, degra_context, _ = clip_model.encode_image(img4clip, control=True)
                                else:
                                    image_context, degra_context = clip_model.encode_image(img4clip, control=True)
                            elif hasattr(clip_model, 'visual_controller') and hasattr(clip_model.visual_controller, 'forward'):
                                image_context, degra_context = clip_model.visual_controller(img4clip, control=True)
                            else:
                                logger.warning("Could not find appropriate encode_image method")
                                continue
                                
                            image_context = image_context.float()
                            degra_context = degra_context.float()
                            
                            noisy_state = sde.noise_state(LQ)
                            model.feed_data(noisy_state, LQ, GT, text_context=degra_context, image_context=image_context)
                            model.test(sde)
                            visuals = model.get_current_visuals()
                            
                            output = util.tensor2img(visuals["Output"].squeeze())
                            gt_img = util.tensor2img(GT.squeeze())
                            
                            avg_psnr_no_sae += util.calculate_psnr(output, gt_img)
                            idx += 1
                    
                    avg_psnr_no_sae /= idx
                    
                    # Restore original SAE mode
                    clip_model.use_sparse_autoencoder = original_sae_mode
                    
                    # Log comparison
                    logger.info(f"PSNR comparison - With SAE: {avg_psnr:.6f}, Without SAE: {avg_psnr_no_sae:.6f}")
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        tb_logger.add_scalar("psnr_no_sae", avg_psnr_no_sae, current_step)
                        tb_logger.add_scalar("psnr_diff", avg_psnr - avg_psnr_no_sae, current_step)
                        
                    # Add feature analysis right here (after the PSNR comparison section)
                    logger.info("Analyzing sparse autoencoder features...")
                    try:
                        os.makedirs('feature_analysis', exist_ok=True)
                        analysis_dir = os.path.join('feature_analysis', f'step_{current_step}')
                            
                        # Analyze sparse features
                        tracker, feature_analysis = analyze_sparse_features(
                            clip_model, 
                            val_loader, 
                            device=device,
                            num_steps=10,
                            save_dir=analysis_dir
                        )
                            
                        # Print summary of most consistent features
                        logger.info(f"=== Feature Analysis at Step {current_step} ===")
                        for feature_idx, analysis in sorted(
                            feature_analysis.items(), 
                            key=lambda x: x[1]['consistency']['degradation_type']['dominant_percentage'] 
                                if 'degradation_type' in x[1]['consistency'] else 0,
                            reverse=True
                        )[:10]:  # Top 10 most consistent features
                            if 'degradation_type' in analysis['consistency']:
                                consistency = analysis['consistency']['degradation_type']
                                logger.info(f"Feature {feature_idx}: {consistency['dominant_percentage']:.1f}% consistently activated by '{consistency['dominant_value']}'")
                    except Exception as e:
                        logger.error(f"Error in sparse feature analysis: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
            if error.value:
                sys.exit(0)
            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()

if __name__ == "__main__":
    main()