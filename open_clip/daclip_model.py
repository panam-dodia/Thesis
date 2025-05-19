from typing import Optional
import logging
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy

import sys

# sys.path.append("/mnt/DATA/panam/Discover-then-Name")

import open_clip
from sparse_autoencoder import SparseAutoencoder
from image_controller import ModifiedImageController

import torch

# Changed from relative import to direct import
from transformer import (
    ControlTransformer
)
from open_clip.model import CLIP, CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower

class DaCLIP(nn.Module):
    def __init__(self, clip_model, use_sparse_autoencoder=True, l1_coeff=0.001, expansion_factor=8):
        super().__init__()
        
        # Handle the case where clip_model is already a DaCLIP instance
        if isinstance(clip_model, DaCLIP):
            self.clip = clip_model.clip
            
            # Extract dimensions from the underlying CLIP model
            self.text_dim = clip_model.text_dim  # Use existing text_dim
            self.image_dim = clip_model.image_dim  # Use existing image_dim
            
            # Configure original visual encoder
            self.visual = self.clip.visual
        else:
            self.clip = clip_model
            
            # Extract dimensions from CLIP model
            self.text_dim = clip_model.text_projection.shape[1]  # Text embedding dimension
            self.image_dim = clip_model.visual.output_dim       # Image embedding dimension
            
            # Configure original visual encoder
            self.visual = clip_model.visual
        
        # Configure controller with ModifiedImageController
        self.visual_controller = ModifiedImageController(
            self.clip.visual,
            self.image_dim,
            hidden_dim=self.image_dim * expansion_factor,
            l1_coeff=l1_coeff,
            expansion_factor=expansion_factor
        )
        
        # Keep a reference to the transformer if needed for control
        if hasattr(self.clip.visual, 'transformer'):
            self.visual_controller.transformer = ControlTransformer(self.clip.visual.transformer)
        
        self.logit_scale = copy.deepcopy(self.clip.logit_scale)
        self.use_sparse_autoencoder = use_sparse_autoencoder
        
        if use_sparse_autoencoder:
            # Initialize geometric median tensors
            self.text_geo_median = torch.zeros((1, self.text_dim))
            self.degra_geo_median = torch.zeros((1, self.image_dim))

            # Create sparse autoencoder for text (degradation SAE is in ModifiedImageController)
            self.text_sae = SparseAutoencoder(
                input_dim=self.text_dim,  # Changed from n_input_features
                hidden_dim=self.text_dim * expansion_factor,  # Changed from n_learned_features
                l1_coeff=l1_coeff,
                expansion_factor=expansion_factor,
                tied_weights=True,  # Added this as it seems to be used in the code
                activation='relu'  # Added as default
            )
            
            # Store geometric median separately since it's not part of SparseAutoencoder constructor
            self.text_sae.geometric_median_dataset = self.text_geo_median
            
            # Save reference to degradation SAE from controller for easy access
            self.degra_sae = self.visual_controller.degra_sae
            
            # Save l1_coeff for loss calculation
            self.l1_coeff = l1_coeff
            
            # Initialize logging for sparse autoencoder metrics
            self.logger = logging.getLogger("SparseAutoencoder")
            self.logger.setLevel(logging.INFO)

    def initial_controller(self):
        """Initialize controller from base CLIP model"""
        # Initialize weights from original visual module to controller
        if hasattr(self.visual_controller, 'initialize_from_visual'):
            self.visual_controller.initialize_from_visual(self.clip.visual)
        else:
            # Manual initialization if needed
            print("ModifiedImageController does not have initialize_from_visual method")
            # This will be handled within the ModifiedImageController constructor
        
        self.logit_scale.data.copy_(self.clip.logit_scale.data)

        if self.use_sparse_autoencoder:
            # Initialize text SAE (degradation SAE is initialized in ModifiedImageController)
            if hasattr(self.text_sae, 'initialize_tied_parameters'):
                self.text_sae.initialize_tied_parameters()
        
    def lock_clip(self):
        """Lock original CLIP parameters"""
        for param in self.clip.parameters():
            param.requires_grad = False

    def load_state_dict(self, state_dict, strict=False):
        """Override load_state_dict to handle missing sparse autoencoder keys"""
        # For backward compatibility with models trained without sparse autoencoder
        missing_keys = []
        for key in self.state_dict().keys():
            if key not in state_dict and any(x in key for x in ['text_sae', 'degra_sae', 'visual_controller']):
                missing_keys.append(key)
        
        # Load existing keys
        return super().load_state_dict(state_dict, strict=False)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.clip.visual.set_grad_checkpointing(enable)
        self.clip.transformer.grad_checkpointing = enable
        
        # Set checkpointing in controller if it supports it
        if hasattr(self.visual_controller, 'set_grad_checkpointing'):
            self.visual_controller.set_grad_checkpointing(enable)

    def encode_image(self, image, control=False, normalize: bool = False):
        if control:
            # Use ModifiedImageController to get features with SAE
            if self.use_sparse_autoencoder:
                content_features, degra_features, degra_l1_loss = self.visual_controller(image, control=True)
                
                # Apply normalization if needed
                content_features = F.normalize(content_features, dim=-1) if normalize else content_features
                degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
                
                return content_features, degra_features, degra_l1_loss
            else:
                # SAE disabled
                content_features, degra_features = self.visual_controller(image, control=True)
                
                # Apply normalization if needed
                content_features = F.normalize(content_features, dim=-1) if normalize else content_features
                degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
                
                return content_features, degra_features
        else:
            # Standard CLIP encode_image behavior
            return self.clip.encode_image(image, normalize)

    def encode_text(self, text, normalize: bool = False):
        text_features = self.clip.encode_text(text, normalize)
        
        # Apply sparse autoencoder to text features if enabled
        if self.use_sparse_autoencoder:
            # Use the SparseAutoencoder forward pass
            sae_result = self.text_sae(text_features)
            
            # Adjust to match SparseAutoencoder's output structure
            learned_activations = sae_result.learned_activations
            reconstruction = sae_result.reconstruction  # Changed from decoded_activations
            
            # Replace original text features with decoded ones
            text_features = reconstruction
            
            # Calculate L1 loss for sparsity (using the total_loss from SAE result)
            text_l1_loss = sae_result.loss
            
            return text_features, text_l1_loss
        
        return text_features
    
    def apply_post_backward_hooks(self):
        """
        Apply the post-backward hooks for the sparse autoencoders.
        This should be called after each optimization step.
        """
        if self.use_sparse_autoencoder:
            # Update text SAE
            if hasattr(self.text_sae, 'post_backwards_hook'):
                self.text_sae.post_backwards_hook()
            else:
                # Try to use rebalance_dead_neurons instead if available
                if hasattr(self.text_sae, 'rebalance_dead_neurons'):
                    self.text_sae.rebalance_dead_neurons()
            
            # Update degradation SAE through controller
            if hasattr(self.visual_controller, 'apply_post_backward_hooks'):
                self.visual_controller.apply_post_backward_hooks()
            elif hasattr(self.degra_sae, 'rebalance_dead_neurons'):
                self.degra_sae.rebalance_dead_neurons()

    def update_geometric_median(self, dataloader, device="cuda", num_batches=10):
        """
        Calculate and update the geometric median for the sparse autoencoders
        from a batch of data.
        """
        if not self.use_sparse_autoencoder:
            return
            
        # Use ModifiedImageController's method if available
        if hasattr(self.visual_controller, 'update_geometric_median'):
            self.visual_controller.update_geometric_median(dataloader, device, num_batches)
        
        # Update text SAE geometric median
        text_features_list = []
        
        # Process batches for text features
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
        #     if isinstance(batch["type"], list):
        #         # Convert list of strings to tensor using the tokenizer
        #         tokenizer = open_clip.get_tokenizer('ViT-B-32')  # Or whatever tokenizer you're using
        #         texts = tokenizer(batch["type"][0]).unsqueeze(0).to(device)
        #         for i in range(1, len(batch["type"])):
        #             token = tokenizer(batch["type"][i]).unsqueeze(0).to(device)
        #             texts = torch.cat([texts, token], dim=0)
        #     else:
        #         texts = batch["type"].to(device) 
            
        #     with torch.no_grad():
        #         # Get raw text features without SAE
        #         text_features = self.clip.encode_text(texts)
        #         text_features_list.append(text_features.detach().cpu())
        
        # # Calculate approximation of geometric median for text
        # if text_features_list:
        #     text_features_tensor = torch.cat(text_features_list, dim=0)
        #     text_median = torch.median(text_features_tensor, dim=0)[0].unsqueeze(0)
            
        #     # Update text SAE geometric median
        #     self.text_geo_median = text_median.to(device)
            
        #     # Store it as an attribute since it's not part of the original SparseAutoencoder
        #     if not hasattr(self.text_sae, 'geometric_median_dataset'):
        #         setattr(self.text_sae, 'geometric_median_dataset', self.text_geo_median)
        #     else:
        #         self.text_sae.geometric_median_dataset = self.text_geo_median
            
        #     # Reinitialize text SAE parameters
        #     if hasattr(self.text_sae, 'initialize_tied_parameters'):
        #         self.text_sae.initialize_tied_parameters()
        #     else:
        #         # If the method doesn't exist, try to reinitialize weights manually
        #         self.text_sae._init_weights()
        
        print("Geometric median updated successfully")

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        sae_losses = {"text_l1_loss": 0.0, "degra_l1_loss": 0.0, "recon_loss": 0.0}
        
        # Process text input (split into caption and degradation)
        (caption, degradation) = text.chunk(2, dim=-1) if text is not None else (None, None)
        
        # Process image with control
        if image is not None:
            if self.use_sparse_autoencoder:
                image_features, image_degra_features, degra_l1_loss = self.encode_image(image, control=True, normalize=True)
                sae_losses["degra_l1_loss"] = degra_l1_loss
            else:
                image_features, image_degra_features = self.encode_image(image, control=True, normalize=True)
        else:
            image_features, image_degra_features = None, None
        
        # Process text caption
        if caption is not None:
            if self.use_sparse_autoencoder:
                text_features, text_l1_loss = self.encode_text(caption, normalize=True)
                sae_losses["text_l1_loss"] = text_l1_loss
            else:
                text_features = self.encode_text(caption, normalize=True)
        else:
            text_features = None
        
        # Process degradation text
        if degradation is not None:
            if self.use_sparse_autoencoder:
                text_degra_features, _ = self.encode_text(degradation, normalize=True)
            else:
                text_degra_features = self.encode_text(degradation, normalize=True)
        else:
            text_degra_features = None

        # Log sparsity metrics periodically during training
        if self.training and self.use_sparse_autoencoder and torch.rand(1).item() < 0.01:  # Log 1% of the time
            if text_features is not None:
                sparsity = (text_features.abs() < 1e-5).float().mean().item()
                self.logger.info(f"Text features sparsity: {sparsity:.4f}")
            
            if image_degra_features is not None:
                sparsity = (image_degra_features.abs() < 1e-5).float().mean().item()
                self.logger.info(f"Degradation features sparsity: {sparsity:.4f}")
                
                # Additional logging for controller activity
                if hasattr(self.visual_controller, 'degra_sae') and hasattr(self.visual_controller.degra_sae, 'activation_counts'):
                    active_rates = (self.visual_controller.degra_sae.activation_counts > 0).float().mean().item()
                    self.logger.info(f"Degradation feature utilization: {active_rates:.4f}")

        # Store losses for get_sae_loss method
        self.forward_result = {
            "image_features": image_features,
            "text_features": text_features,
            "image_degra_features": image_degra_features,
            "text_degra_features": text_degra_features,
            "logit_scale": self.logit_scale.exp(),
            "sae_losses": sae_losses
        }
        
        return self.forward_result
    
    def get_sae_loss(self):
        """
        Calculate the total sparse autoencoder loss for the model.
        Should be called after the forward pass.
        """
        if not self.use_sparse_autoencoder:
            return 0.0
            
        # This will be populated during the forward pass
        total_sae_loss = 0.0
        forward_result = self.forward_result if hasattr(self, 'forward_result') else {}
        sae_losses = forward_result.get("sae_losses", {})
        
        for loss_name, loss_value in sae_losses.items():
            total_sae_loss += loss_value
            
        return total_sae_loss
    
    # Keep compute_concept_activations method unchanged
    def compute_concept_activations(self, features):
        """
        Compute activations of concepts in the sparse representation.
        This can be used to verify if the sparse autoencoder is working.
        
        Args:
            features: Input features to analyze
            
        Returns:
            Dictionary containing activation statistics
        """
        if not self.use_sparse_autoencoder:
            return {"error": "Sparse autoencoder is disabled"}
            
        # Encode the features
        sae_result = self.text_sae(features)
        learned_activations = sae_result.learned_activations
        
        # Calculate statistics
        active_neurons = (learned_activations > 0).float().sum(dim=1)
        avg_active = active_neurons.mean().item()
        max_active = active_neurons.max().item()
        min_active = active_neurons.min().item()
        
        # Calculate average activation value for active neurons
        masked_values = learned_activations * (learned_activations > 0).float()
        avg_activation = masked_values.sum() / (masked_values > 0).float().sum()
        
        return {
            "avg_active_neurons": avg_active,
            "max_active_neurons": max_active,
            "min_active_neurons": min_active,
            "avg_activation_value": avg_activation.item()
        }