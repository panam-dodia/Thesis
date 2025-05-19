import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
# sys.path.append("/mnt/DATA/panam/Discover-then-Name")

from sparse_autoencoder import SparseAutoencoder

class ModifiedImageController(nn.Module):
    def __init__(self, image_encoder, image_dim, hidden_dim=None, expansion_factor=8, l1_coeff=0.001):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_dim = image_dim
        
        # Create a copy of the image encoder for degradation features
        self.copy_encoder = copy.deepcopy(image_encoder)
        
        # Determine hidden dimension
        if hidden_dim is None:
            hidden_dim = image_dim * expansion_factor
        
        # Create or get the transformer from original encoder
        if hasattr(image_encoder, 'transformer'):
            self.transformer = copy.deepcopy(image_encoder.transformer)
        
        # Initialize geometric median 
        self.register_buffer('degra_geo_median', torch.zeros((1, image_dim)))
        
        # Add sparse autoencoder for degradation features
        self.degra_sae = SparseAutoencoder(
            input_dim=image_dim,
            hidden_dim=hidden_dim,
            l1_coeff=l1_coeff,
            )
        
        # Flag to enable/disable sparse autoencoder
        self.use_sparse_autoencoder = True
        self.l1_coeff = l1_coeff
    
    def initialize_from_visual(self, visual):
        """Initialize controller weights from original visual module"""
        # Copy parameters that don't include transformer
        for (name_v, param_v), (name_c, param_c) in zip(
            [(n, p) for n, p in visual.named_parameters() if 'transformer' not in n],
            [(n, p) for n, p in self.copy_encoder.named_parameters() if 'transformer' not in n]
        ):
            param_c.data.copy_(param_v.data)
        
        # Copy transformer parameters if present
        if hasattr(visual, 'transformer') and hasattr(self, 'transformer'):
            for param_v, param_c in zip(visual.transformer.parameters(), self.transformer.parameters()):
                param_c.data.copy_(param_v.data)
    
    def forward(self, x, control=False, output_hiddens=False):
        # Get original encoder features
        content_features = self.image_encoder(x)
        
        if not control:
            return content_features
            
        # Get degradation features from controller copy
        degradation_features = self.copy_encoder(x)
        
        # Process through sparse autoencoder if enabled
        if self.use_sparse_autoencoder:
            # Get sparse representation
            sae_result = self.degra_sae(degradation_features)
            learned_activations = sae_result.learned_activations
            decoded_activations = sae_result.decoded_activations
            
            # L1 sparsity loss
            l1_loss = self.l1_coeff * learned_activations.abs().mean()
            
            if output_hiddens:
                return degradation_features, learned_activations
                
            # Return decoded_activations instead of learned_activations
            # This is the key change - it will match the expected dimension
            return content_features, decoded_activations, l1_loss
        else:
            # Original behavior without sparse autoencoder
            if output_hiddens:
                return degradation_features, None
                
            return content_features, degradation_features
    
    def visual_control(self, x, output_hiddens=False):
        """Compatibility method for validation code"""
        degradation_features = self.copy_encoder(x)
        
        if self.use_sparse_autoencoder and not output_hiddens:
            # Get sparse representation
            sae_result = self.degra_sae(degradation_features)
            learned_activations = sae_result.learned_activations
            return degradation_features, learned_activations
        
        return degradation_features, None
            
    def update_geometric_median(self, dataloader, device=None, num_batches=10):
        """Initialize the geometric median of features for better initialization"""
        if device is None:
            device = next(self.parameters()).device
            
        all_features = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                imgs = batch["LQ_clip"].to(device)
                features = self.copy_encoder(imgs)
                all_features.append(features.detach().cpu())
                
        if all_features:
            features_tensor = torch.cat(all_features, dim=0)
            # Use median as a robust estimator for geometric median
            median_features = torch.median(features_tensor, dim=0)[0].unsqueeze(0)
            
            # Update geometric median
            self.degra_geo_median = median_features.to(device)
            
            # Update SAE geometric median
            self.degra_sae.geometric_median_dataset = self.degra_geo_median
            
            # Reinitialize tied parameters
            if hasattr(self.degra_sae, 'initialize_tied_parameters'):
                self.degra_sae.initialize_tied_parameters()
    
    def apply_post_backward_hooks(self):
        """Apply post-backward hooks for SAE maintenance"""
        if hasattr(self.degra_sae, 'post_backwards_hook'):
            self.degra_sae.post_backwards_hook()