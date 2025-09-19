#!/usr/bin/env python3
"""
Feature extraction for thesis - using original DA-CLIP as baseline
"""
import os
import torch
import yaml
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import gc
import sys
import importlib.util
from PIL import Image
from datetime import datetime

# Add paths
sys.path.insert(0, "../../")
sys.path.insert(0, "/mnt/DATA/panam/daclip-uir/universal-image-restoration/open_clip")
sys.path.insert(0, "/mnt/DATA/panam/daclip-uir/universal-image-restoration")
sys.path.insert(0, "/mnt/DATA/panam/daclip-uir/universal-image-restoration/config/daclip-sde")

import open_clip
import options as option
from data import create_dataset, create_dataloader
from models import create_model

# Import SDE utils
module_path = '/mnt/DATA/panam/daclip-uir/universal-image-restoration/utils/sde_utils.py'
spec = importlib.util.spec_from_file_location('sde_utils', module_path)
sde_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sde_utils)
IRSDE = sde_utils.IRSDE

class ThesisFeatureExtractor:
    def __init__(self, use_original_baseline=True, extract_images=False, max_samples_per_deg=None):
        """
        Args:
            use_original_baseline: Use original DA-CLIP (universal-ir.pth) instead of your trained model
            extract_images: Whether to save images
            max_samples_per_deg: Limit samples per degradation
        """
        # Load config
        self.opt = option.parse('./options/test.yml', is_train=False)
        self.opt = option.dict_to_nonedict(self.opt)
        
        # IMPORTANT: Use original model for baseline comparison
        if use_original_baseline:
            print("⚠️ Using ORIGINAL DA-CLIP model (universal-ir.pth) for proper baseline comparison")
            self.opt['path']['pretrain_model_G'] = '/mnt/DATA/panam/daclip-uir/pretrained/universal-ir-mix.pth'
        else:
            print("Using your trained model (lastest_EMA.pth)")
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Output directory with thesis prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = "original_baseline" if use_original_baseline else "trained_model"
        self.output_dir = Path(f'./thesis_features_full_new_{model_type}_{timestamp}')
        
        self.target_degradations = ['rainy', 'snowy', 'raindrop', 'low-light', 'shadowed', 'inpainting']
        
        self.extract_images = extract_images
        self.max_samples_per_deg = max_samples_per_deg
        
        # Statistics tracking
        self.stats = {
            'train': {deg: 0 for deg in self.target_degradations},
            'val': {deg: 0 for deg in self.target_degradations}
        }
        
        self.load_models()
    
    def load_models(self):
        """Load DA-CLIP and diffusion models"""
        print(f"Loading models...")
        print(f"Diffusion model: {self.opt['path']['pretrain_model_G']}")
        print(f"CLIP model: {self.opt['path']['daclip']}")
        
        # Verify model exists
        if not Path(self.opt['path']['pretrain_model_G']).exists():
            raise FileNotFoundError(f"Model not found: {self.opt['path']['pretrain_model_G']}")
        
        # Load DA-CLIP
        base_clip, _ = open_clip.create_model_from_pretrained(
            'daclip_ViT-B-32', 
            pretrained=self.opt['path']['daclip']
        )
        
        from daclip_model import DaCLIP
        
        if not hasattr(base_clip, 'visual_controller'):
            self.clip_model = DaCLIP(base_clip, use_sparse_autoencoder=False)
            if hasattr(self.clip_model, 'initial_controller'):
                self.clip_model.initial_controller()
        else:
            self.clip_model = base_clip
            
        self.clip_model.use_sparse_autoencoder = False
        self.clip_model = self.clip_model.to(self.device).eval()
        
        # Load diffusion model
        self.diffusion_model = create_model(self.opt)
        
        if hasattr(self.diffusion_model, 'model'):
            self.diffusion_model.model = self.diffusion_model.model.to(self.device)
        
        if hasattr(self.diffusion_model, 'netG'):
            self.diffusion_model.netG = self.diffusion_model.netG.to(self.device)
            
        # Initialize SDE
        self.sde = IRSDE(
            max_sigma=self.opt["sde"]["max_sigma"],
            T=self.opt["sde"]["T"],
            schedule=self.opt["sde"]["schedule"],
            eps=self.opt["sde"]["eps"],
            device=self.device
        )
        self.sde.set_model(self.diffusion_model.model)
        
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        print("✅ Models loaded successfully")
    
    def preprocess_for_clip(self, img):
        """Preprocess image for CLIP"""
        if img.max() > 2.0:
            img = img / 255.0
        if img.max() <= 1.0 and img.min() >= 0.0:
            img = img * 2.0 - 1.0
        return img
    
    def ensure_4d_tensor(self, tensor):
        """Ensure tensor is 4D"""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor
    
    def process_batch(self, batch_data, phase, save_images=False):
        """Process a single batch and extract features"""
        deg_type = batch_data["type"][0] if isinstance(batch_data["type"], list) else batch_data["type"]
        
        # Skip if not target degradation
        if deg_type not in self.target_degradations:
            return None
        
        # Check sample limit
        if self.max_samples_per_deg and self.stats[phase][deg_type] >= self.max_samples_per_deg:
            return None
        
        try:
            # Move to device
            lq_image = batch_data["LQ"].to(self.device)
            img4clip = batch_data["LQ_clip"].to(self.device)
            
            # Store GT if available for comparison
            has_gt = "GT" in batch_data
            if has_gt:
                gt_image = batch_data["GT"].to(self.device)
                gt_image = self.ensure_4d_tensor(gt_image)
            
            # Ensure 4D
            lq_image = self.ensure_4d_tensor(lq_image)
            img4clip = self.ensure_4d_tensor(img4clip)
            
            # Preprocess for CLIP
            img4clip = self.preprocess_for_clip(img4clip)
            img4clip = F.interpolate(img4clip, size=(224, 224), mode='bilinear')
            
            with torch.no_grad():
                # 1. Extract pre-diffusion features (after image encoder)
                image_result = self.clip_model.encode_image(img4clip, control=False, normalize=False)
                if isinstance(image_result, tuple):
                    features_pre = image_result[0]
                else:
                    features_pre = image_result
                features_pre = F.normalize(features_pre.float(), dim=-1)
                
                # 2. Get conditioning for diffusion
                deg_token = self.tokenizer(deg_type).to(self.device)
                text_features = self.clip_model.encode_text(deg_token, normalize=False)
                if isinstance(text_features, tuple):
                    text_features = text_features[0]
                text_features = F.normalize(text_features.float(), dim=-1)
                
                controller_result = self.clip_model.encode_image(img4clip, control=True, normalize=False)
                if isinstance(controller_result, tuple) and len(controller_result) >= 2:
                    content_features, controller_features = controller_result[:2]
                else:
                    controller_features = features_pre
                    content_features = features_pre
                
                content_features = F.normalize(content_features.float(), dim=-1)
                controller_features = F.normalize(controller_features.float(), dim=-1)
                
                # 3. Run diffusion restoration
                if lq_image.max() > 1.0:
                    lq_image = torch.clamp(lq_image / 255.0, 0.0, 1.0)
                
                timesteps, states = self.sde.generate_random_states(x0=lq_image, mu=lq_image)
                states = states.to(self.device)
                
                self.diffusion_model.feed_data(
                    states, lq_image, lq_image,
                    text_context=controller_features,
                    image_context=content_features
                )
                
                self.diffusion_model.test(self.sde)
                visuals = self.diffusion_model.get_current_visuals()
                restored = visuals["Output"]
                
                restored = self.ensure_4d_tensor(restored)
                restored = restored.to(self.device)
                
                # 4. Extract post-diffusion features (before image decoder)
                restored = torch.clamp(restored, 0.0, 1.0)
                restored_clip = self.preprocess_for_clip(restored)
                restored_clip = F.interpolate(restored_clip, size=(224, 224), mode='bilinear')
                restored_clip = restored_clip.to(self.device)
                
                restored_result = self.clip_model.encode_image(restored_clip, control=False, normalize=False)
                if isinstance(restored_result, tuple):
                    features_post = restored_result[0]
                else:
                    features_post = restored_result
                features_post = F.normalize(features_post.float(), dim=-1)
                
                # 5. Extract GT features if available
                features_gt = None
                if has_gt:
                    gt_clip = self.preprocess_for_clip(gt_image)
                    gt_clip = F.interpolate(gt_clip, size=(224, 224), mode='bilinear')
                    gt_result = self.clip_model.encode_image(gt_clip, control=False, normalize=False)
                    if isinstance(gt_result, tuple):
                        features_gt = gt_result[0]
                    else:
                        features_gt = gt_result
                    features_gt = F.normalize(features_gt.float(), dim=-1)
            
            # Prepare output
            idx = self.stats[phase][deg_type]
            
            return {
                'deg_type': deg_type,
                'idx': idx,
                'features_pre': features_pre.cpu().squeeze(),
                'features_post': features_post.cpu().squeeze(),
                'features_gt': features_gt.cpu().squeeze() if features_gt is not None else None,
                'lq_image': lq_image.cpu() if save_images else None,
                'restored': restored.cpu() if save_images else None,
                'gt_image': gt_image.cpu() if (save_images and has_gt) else None
            }
            
        except Exception as e:
            print(f"Error processing {deg_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_features(self, result, phase):
        """Save extracted features to disk"""
        deg_type = result['deg_type']
        idx = result['idx']
        
        # Create directories
        out_dir = self.output_dir / phase / deg_type
        (out_dir / 'pre_diffusion').mkdir(parents=True, exist_ok=True)
        (out_dir / 'post_diffusion').mkdir(parents=True, exist_ok=True)
        
        # Save features
        torch.save(result['features_pre'], 
                  out_dir / 'pre_diffusion' / f'sample_{idx:05d}.pt')
        torch.save(result['features_post'], 
                  out_dir / 'post_diffusion' / f'sample_{idx:05d}.pt')
        
        # Save GT features if available
        if result['features_gt'] is not None:
            (out_dir / 'ground_truth').mkdir(parents=True, exist_ok=True)
            torch.save(result['features_gt'],
                      out_dir / 'ground_truth' / f'sample_{idx:05d}.pt')
        
        # Save images if requested
        if self.extract_images and result['lq_image'] is not None:
            (out_dir / 'input_images').mkdir(parents=True, exist_ok=True)
            (out_dir / 'restored_images').mkdir(parents=True, exist_ok=True)
            
            def tensor_to_pil(tensor):
                tensor = tensor.squeeze().cpu()
                if tensor.dim() == 3:
                    tensor = tensor.permute(1, 2, 0)
                tensor = tensor.numpy()
                tensor = (tensor * 255).astype(np.uint8)
                return Image.fromarray(tensor)
            
            input_img = tensor_to_pil(result['lq_image'])
            restored_img = tensor_to_pil(result['restored'])
            
            input_img.save(out_dir / 'input_images' / f'sample_{idx:05d}.png')
            restored_img.save(out_dir / 'restored_images' / f'sample_{idx:05d}.png')
            
            if result['gt_image'] is not None:
                (out_dir / 'gt_images').mkdir(parents=True, exist_ok=True)
                gt_img = tensor_to_pil(result['gt_image'])
                gt_img.save(out_dir / 'gt_images' / f'sample_{idx:05d}.png')
    
    def extract_dataset(self, phase='train'):
        """Extract features from dataset"""
        print(f"\n{'='*50}")
        print(f"Extracting {phase} dataset features")
        print(f"{'='*50}")
        
        # Update batch size to 1 for extraction
        self.opt['datasets'][phase]['batch_size'] = 1
        dataset = create_dataset(self.opt['datasets'][phase])
        dataloader = create_dataloader(dataset, self.opt['datasets'][phase], self.opt, None)
        
        total_batches = len(dataloader)
        print(f"Total batches to process: {total_batches}")
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Processing {phase}")):
            result = self.process_batch(batch_data, phase, save_images=self.extract_images)
            
            if result is not None:
                self.save_features(result, phase)
                deg_type = result['deg_type']
                self.stats[phase][deg_type] += 1
                
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        print(f"\n{phase.upper()} extraction complete:")
        for deg, count in self.stats[phase].items():
            if count > 0:
                print(f"  {deg}: {count} samples")
    
    def save_metadata(self):
        """Save extraction metadata"""
        metadata = {
            'extraction_date': datetime.now().isoformat(),
            'statistics': self.stats,
            'model_paths': {
                'daclip': self.opt['path']['daclip'],
                'diffusion': self.opt['path']['pretrain_model_G']
            },
            'device': str(self.device),
            'target_degradations': self.target_degradations,
            'output_directory': str(self.output_dir)
        }
        
        with open(self.output_dir / 'thesis_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Metadata saved to {self.output_dir / 'thesis_metadata_full_new.json'}")
    
    def run(self):
        """Run extraction pipeline"""
        print(f"Starting THESIS feature extraction")
        print(f"Output directory: {self.output_dir}")
        print(f"Extract images: {self.extract_images}")
        print(f"Max samples per degradation: {self.max_samples_per_deg or 'All'}")
        
        # Extract training data
        self.extract_dataset('train')
        
        # Extract validation data
        self.extract_dataset('val')
        
        # Save metadata
        self.save_metadata()
        
        print(f"\n{'='*50}")
        print("THESIS FEATURE EXTRACTION COMPLETE")
        print(f"{'='*50}")
        print(f"Total samples extracted:")
        print(f"Train: {sum(self.stats['train'].values())}")
        print(f"Val: {sum(self.stats['val'].values())}")
        print(f"\nFeatures saved to: {self.output_dir}")

if __name__ == "__main__":    
    # Test with limited samples first
    extractor = ThesisFeatureExtractor(
        use_original_baseline=True,
        extract_images=False,  # Save images for verification
        max_samples_per_deg=None  # Start with 10 samples for testing
    )
    
    extractor.run()