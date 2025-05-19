import os
import os.path as osp
import shutil
import logging
import datetime
import numpy as np
import torch
import cv2
import math

def mkdir(path):
    """Make a directory if it doesn't already exist."""
    if not osp.exists(path):
        os.makedirs(path)

def mkdir_and_rename(path):
    """Rename the directory if it already exists, then create a new one."""
    if osp.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print(f'Path [{path}] already exists. Rename it to [{new_name}]')
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)

def mkdirs(paths):
    """Make directories."""
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def get_timestamp():
    """Get the current timestamp formatted as a string."""
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    """Set up logger."""
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                 datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger

def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dict2str(opt, indent_level=1):
    """Dict to string for printing options."""
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':[\n'
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg

def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)."""
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert tensor to image."""
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    img_np = tensor.numpy()
    img_np = img_np * 255.0
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    img_np = img_np.astype(out_type)
    return img_np

def save_img(img, img_path, mode='RGB'):
    """Save image."""
    cv2.imwrite(img_path, img)