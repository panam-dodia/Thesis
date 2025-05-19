from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from collections import namedtuple

# Update the field names in the namedtuple
SparseAutoencoderOutput = namedtuple('SparseAutoencoderOutput', 
                                     ['decoded_activations', 'learned_activations', 'loss'])

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, l1_coeff=0.001, expansion_factor=8, tied_weights=False, activation='relu'):
        super().__init__()
        
        # If hidden_dim not specified, use expansion factor
        if hidden_dim is None:
            hidden_dim = input_dim * expansion_factor
        
        self.input_dim = input_dim
        self.n_learned_features = hidden_dim
        self.l1_coeff = l1_coeff
        self.tied_weights = tied_weights
        # self.geometric_median_dataset = geometric_median_dataset
        
        # Encoder (input dimension to hidden dimension)
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder (hidden dimension back to input dimension)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Tie weights if specified
        if tied_weights:
            self.decoder.weight = nn.Parameter(self.encoder.weight.t())
            
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
            
        # Initialize with better defaults
        self._init_weights()
        
        # Cache for dead neurons tracking
        self.register_buffer('activation_counts', torch.zeros(hidden_dim))
        self.register_buffer('activation_history', torch.zeros(hidden_dim, 100))
        self.history_idx = 0
    
    def _init_weights(self):
        # Initialize with scaled orthogonal weights
        nn.init.orthogonal_(self.encoder.weight, gain=1.0)
        nn.init.zeros_(self.encoder.bias)
        
        if not self.tied_weights:
            nn.init.orthogonal_(self.decoder.weight, gain=1.0)
            nn.init.zeros_(self.decoder.bias)
    
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        
        # Activate with chosen activation function
        h_activated = self.activation(h)
        
        # Apply explicit sparsity
        if self.training:
            # Use a small threshold to count activations
            active_neurons = (h_activated > 0.1).float()
            self.activation_counts += active_neurons.sum(dim=0)
            
            # Store recent activation pattern
            self.activation_history[:, self.history_idx % 100] = active_neurons.mean(dim=0)
            self.history_idx += 1
            
            # L1 sparsity loss - apply higher coefficient to less active neurons to balance use
            recent_activations = self.activation_history.mean(dim=1)
            l1_scaling = 1.0 / (recent_activations + 0.01)
            l1_scaling = l1_scaling / l1_scaling.mean()  # Normalize scaling factors
            
            weighted_l1 = self.l1_coeff * (torch.abs(h_activated) * l1_scaling.unsqueeze(0)).mean()
        else:
            weighted_l1 = self.l1_coeff * torch.abs(h_activated).mean()
            
        # Decode
        reconstructed = self.decoder(h_activated)
        
        # Calculate reconstruction loss
        mse_loss = F.mse_loss(reconstructed, x)
        
        # Total loss
        total_loss = mse_loss + weighted_l1
        
        # Return using the namedtuple with the correct field name
        return SparseAutoencoderOutput(
            decoded_activations=reconstructed,  # Changed from reconstruction to decoded_activations
            learned_activations=h_activated,
            loss=total_loss
        )
    
    def encode(self, x):
        """Only encode the input and return the sparse hidden representation"""
        hidden = self.encoder(x)
        return self.activation(hidden)  # Fixed the variable name from h to hidden

    def decode(self, h):
        """Decode from the hidden representation back to the input space"""
        return self.decoder(h)
    
    def rebalance_dead_neurons(self, threshold=0.01):
        """Replace dead neurons with copies of the most active ones"""
        if not self.training:
            return
            
        with torch.no_grad():
            # Calculate activation frequency
            total_samples = self.history_idx if self.history_idx < 100 else 100
            if total_samples == 0:
                return
                
            activation_frequency = self.activation_history[:, :total_samples].mean(dim=1)
            
            # Identify dead neurons (consistently inactive)
            dead_mask = activation_frequency < threshold
            dead_count = dead_mask.sum().item()
            
            # Only rebalance if there are dead neurons
            if dead_count > 0:
                # Find most active neurons
                _, top_indices = torch.topk(activation_frequency, k=dead_count)
                dead_indices = torch.where(dead_mask)[0]
                
                # Copy weights from active to dead neurons with small perturbations
                for i, dead_idx in enumerate(dead_indices):
                    active_idx = top_indices[i % len(top_indices)]
                    
                    # Copy encoder weights with perturbation
                    self.encoder.weight[dead_idx] = self.encoder.weight[active_idx] * (0.9 + 0.2 * torch.rand_like(self.encoder.weight[active_idx]))
                    
                    # Copy bias
                    self.encoder.bias[dead_idx] = self.encoder.bias[active_idx] * (0.9 + 0.2 * torch.rand_like(self.encoder.bias[active_idx]))
                    
                    # Copy decoder weights if not using tied weights
                    if not self.tied_weights:
                        self.decoder.weight[:, dead_idx] = self.decoder.weight[:, active_idx] * (0.9 + 0.2 * torch.rand_like(self.decoder.weight[:, active_idx]))
                        
                # Reset activation counts for rebalanced neurons
                self.activation_counts[dead_mask] = 0
                self.activation_history[dead_mask] = 0