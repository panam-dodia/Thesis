# sae_validation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json

def check_sparsity(model, images):
    """Check sparsity of sparse autoencoder features"""
    if not hasattr(model, 'degra_sae'):
        if hasattr(model, 'visual_controller') and hasattr(model.visual_controller, 'degra_sae'):
            # Access through visual_controller
            sae = model.visual_controller.degra_sae
        else:
            return {"error": "No sparse autoencoder found"}
    else:
        # Direct access
        sae = model.degra_sae
        
    with torch.no_grad():
        # Get degradation features - Fix: use visual_controller instead of visual_control
        if hasattr(model, 'visual_controller') and hasattr(model.visual_controller, 'forward'):
            # Use visual_controller's forward method
            if hasattr(model.visual_controller, 'output_hiddens'):
                degra_features, _ = model.visual_controller(images, output_hiddens=True)
            else:
                # Fall back to calling forward without output_hiddens
                print("Using alternative method to get features")
                result = model.visual_controller(images, control=True)
                if isinstance(result, tuple) and len(result) >= 2:
                    _, degra_features = result[:2]
                else:
                    print(f"Unexpected result type from visual_controller: {type(result)}")
                    return {"error": "Could not get features from model"}
        else:
            # Fallback to encode_image
            print("Falling back to encode_image method")
            result = model.encode_image(images, control=True)
            if isinstance(result, tuple) and len(result) >= 2:
                _, degra_features = result[:2]
            else:
                print(f"Unexpected result type from encode_image: {type(result)}")
                return {"error": "Could not get features from model"}
        
        # Apply SAE directly
        sae_result = sae(degra_features)
        
        # Check what fields are available in the SAE result
        if hasattr(sae_result, 'learned_activations'):
            learned_activations = sae_result.learned_activations
        else:
            # Try to check if it's a tuple or has other possible field names
            print(f"SAE result type: {type(sae_result)}")
            if isinstance(sae_result, tuple) and len(sae_result) >= 2:
                learned_activations = sae_result[1]  # Assuming second element is activations
            else:
                print("Could not get learned_activations from SAE result")
                return {"error": "Could not access learned_activations"}
        
        # Calculate sparsity (using small threshold)
        sparsity = (torch.abs(learned_activations) < 1e-5).float().mean().item()
        
        # Get statistics
        active_count = (torch.abs(learned_activations) >= 1e-5).sum().item()
        total_count = learned_activations.numel()
        
        # Get decoder norms
        if hasattr(sae, 'decoder'):
            decoder_norms = sae.decoder.weight.norm(dim=0)
            min_norm = decoder_norms.min().item()
            max_norm = decoder_norms.max().item()
        else:
            min_norm = max_norm = 0.0
            
        return {
            "sparsity": sparsity,
            "active_count": active_count,
            "total_count": total_count,
            "min_decoder_norm": min_norm,
            "max_decoder_norm": max_norm,
            "features": learned_activations.detach().cpu()  # For visualization
        }

def visualize_feature_activations(features, save_path, max_features=100):
    """Create visualization of feature activations"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Take first batch item if batched
        if features.dim() > 2:
            features = features[0]
        
        # Get feature norms
        feature_norms = torch.norm(features, dim=1)
        
        # Sort by activation strength
        sorted_indices = torch.argsort(feature_norms, descending=True)
        
        # Take top features
        top_indices = sorted_indices[:max_features].cpu().numpy()
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.bar(np.arange(len(top_indices)), feature_norms[top_indices].cpu().numpy())
        plt.title(f"Top {len(top_indices)} Feature Activations")
        plt.xlabel("Feature Index")
        plt.ylabel("Activation Norm")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        # Create sparsity visualization
        plt.figure(figsize=(12, 8))
        activation_matrix = features.abs().cpu().numpy()
        plt.imshow(activation_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='Activation Magnitude')
        plt.title("Feature Activation Pattern")
        plt.xlabel("Feature Index")
        plt.ylabel("Batch Sample")
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_matrix.png'))
        plt.close()
        
        print(f"Visualizations saved to {save_path}")
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

class FeatureActivationTracker:
    def __init__(self, num_features, max_samples_per_feature=100):
        self.num_features = num_features
        self.max_samples_per_feature = max_samples_per_feature
        self.feature_activations = defaultdict(list)
        self.feature_metadata = defaultdict(list)
        
    def add_batch(self, learned_activations, metadata=None):
        """Track activations for a batch of inputs"""
        batch_size = learned_activations.shape[0]
        
        for feature_idx in range(self.num_features):
            feature_acts = learned_activations[:, feature_idx]
            
            # Find samples with strong activations for this feature
            strong_act_mask = feature_acts > 0.1  # Threshold for "strong" activation
            strong_indices = torch.where(strong_act_mask)[0]
            
            for idx in strong_indices:
                if len(self.feature_activations[feature_idx]) < self.max_samples_per_feature:
                    self.feature_activations[feature_idx].append({
                        'activation': feature_acts[idx].item(),
                        'input_idx': idx.item(),
                        'metadata': metadata[idx] if metadata is not None else {}
                    })
    
    def get_top_activations(self, feature_idx, top_k=10):
        """Get the top K activations for a specific feature"""
        feature_data = self.feature_activations[feature_idx]
        sorted_data = sorted(feature_data, key=lambda x: x['activation'], reverse=True)
        return sorted_data[:top_k]
    
    def analyze_consistency(self, feature_idx):
        """Analyze consistency of what activates a specific feature"""
        feature_data = self.feature_activations[feature_idx]
        
        if not feature_data:
            return {"error": "No activations for this feature"}
        
        # Collect all metadata keys
        metadata_values = defaultdict(list)
        for item in feature_data:
            for key, value in item['metadata'].items():
                metadata_values[key].append(value)
        
        # Analyze consistency for each metadata field
        consistency_analysis = {}
        for key, values in metadata_values.items():
            unique_values = set(values)
            value_counts = {val: values.count(val) for val in unique_values}
            total_count = len(values)
            
            consistency_analysis[key] = {
                'unique_values': len(unique_values),
                'dominant_value': max(value_counts, key=value_counts.get),
                'dominant_percentage': (value_counts[max(value_counts, key=value_counts.get)] / total_count) * 100,
                'value_distribution': value_counts
            }
        
        return consistency_analysis

    def visualize_feature_activations(self, feature_idx, save_path=None):
        """Visualize what activates a specific feature"""
        feature_data = self.feature_activations[feature_idx]
        
        if not feature_data:
            print(f"No activations found for feature {feature_idx}")
            return
        
        try:
            # Ensure the directory exists if save_path is provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
            # Extract activations and sort them
            activations = [item['activation'] for item in feature_data]
            sorted_indices = np.argsort(activations)[::-1]
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Activation strength histogram
            ax1.hist(activations, bins=20, edgecolor='black')
            ax1.set_title(f'Feature {feature_idx} Activation Distribution')
            ax1.set_xlabel('Activation Strength')
            ax1.set_ylabel('Count')
            
            # Top activations bar chart
            top_k = min(10, len(feature_data))
            top_activations = [feature_data[idx]['activation'] for idx in sorted_indices[:top_k]]
            top_labels = [f"Input {feature_data[idx]['input_idx']}" for idx in sorted_indices[:top_k]]
            
            ax2.barh(range(top_k), top_activations)
            ax2.set_yticks(range(top_k))
            ax2.set_yticklabels(top_labels)
            ax2.set_xlabel('Activation Strength')
            ax2.set_title(f'Top {top_k} Activations for Feature {feature_idx}')
            ax2.invert_yaxis()  # Highest at top
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
                print(f"Visualization saved to {save_path}")
            else:
                plt.show()
        except Exception as e:
            print(f"Error creating visualization for feature {feature_idx}: {str(e)}")

def analyze_sparse_features(model, dataloader, device, num_steps=10, save_dir='feature_analysis'):
    """Analyze which inputs activate which features"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")
        
        # Determine which SAE to use
        if hasattr(model, 'degra_sae'):
            sae = model.degra_sae
        elif hasattr(model, 'visual_controller') and hasattr(model.visual_controller, 'degra_sae'):
            sae = model.visual_controller.degra_sae
        else:
            print("No sparse autoencoder found")
            return None, {}
        
        # Initialize tracker
        tracker = FeatureActivationTracker(
            num_features=sae.n_learned_features
        )
        
        model.eval()
        step = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if step >= num_steps:
                    break
                    
                images = batch["LQ_clip"].to(device)
                degradation_types = batch["type"]
                
                # Get degradation features - Fix: use visual_controller instead of visual_control
                if hasattr(model, 'visual_controller') and hasattr(model.visual_controller, 'forward'):
                    # Try using output_hiddens parameter first
                    try:
                        degra_features, learned_activations = model.visual_controller(images, output_hiddens=True)
                    except Exception as e:
                        print(f"Error with output_hiddens=True: {str(e)}")
                        # Fall back to regular forward
                        result = model.visual_controller(images, control=True)
                        if isinstance(result, tuple) and len(result) >= 2:
                            _, degra_features = result[:2]
                            # Apply SAE directly
                            sae_result = sae(degra_features)
                            learned_activations = sae_result.learned_activations
                        else:
                            print(f"Unexpected result type: {type(result)}")
                            continue
                else:
                    print("Using encode_image as fallback")
                    # Use encode_image as fallback
                    try:
                        result = model.encode_image(images, control=True)
                        if isinstance(result, tuple) and len(result) >= 2:
                            _, degra_features = result[:2]
                            # Apply SAE directly
                            sae_result = sae(degra_features)
                            learned_activations = sae_result.learned_activations
                        else:
                            print(f"Unexpected result type: {type(result)}")
                            continue
                    except Exception as e:
                        print(f"Error using encode_image: {str(e)}")
                        continue
                
                # Create metadata for each sample
                metadata = []
                for i in range(len(degradation_types)):
                    metadata.append({
                        'degradation_type': degradation_types[i],
                        'batch_idx': step,
                        'sample_idx': i
                    })
                
                # Track activations
                tracker.add_batch(learned_activations, metadata)
                
                step += 1
        
        # Analyze and visualize top features
        active_features = []
        feature_analysis = {}
        
        for feature_idx in range(sae.n_learned_features):
            if feature_idx in tracker.feature_activations and tracker.feature_activations[feature_idx]:
                active_features.append(feature_idx)
                
                # Get top activations
                top_acts = tracker.get_top_activations(feature_idx)
                
                # Analyze consistency
                consistency = tracker.analyze_consistency(feature_idx)
                
                feature_analysis[str(feature_idx)] = {
                    'top_activations': top_acts,
                    'consistency': consistency
                }
                
                # Visualize
                tracker.visualize_feature_activations(
                    feature_idx, 
                    save_path=os.path.join(save_dir, f'feature_{feature_idx}_analysis.png')
                )
        
        # Save analysis results
        with open(os.path.join(save_dir, 'feature_analysis_summary.json'), 'w') as f:
            json.dump(feature_analysis, f, indent=2)
        
        # Create summary visualization
        create_summary_visualization(feature_analysis, save_dir)
        
        print(f"Completed analysis, {len(active_features)} active features found")
        return tracker, feature_analysis
        
    except Exception as e:
        print(f"Error in analyze_sparse_features: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, {}

def create_summary_visualization(feature_analysis, save_dir):
    """Create a summary visualization of feature consistency"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        consistency_scores = []
        feature_indices = []
        degradation_associations = {}
        
        for feature_idx, analysis in feature_analysis.items():
            if 'consistency' in analysis and 'degradation_type' in analysis['consistency']:
                consistency = analysis['consistency']['degradation_type']
                consistency_scores.append(consistency['dominant_percentage'])
                feature_indices.append(int(feature_idx))
                degradation_associations[int(feature_idx)] = consistency['dominant_value']
        
        if not consistency_scores:
            print("No consistency data available for visualization")
            return
        
        # Plot consistency scores
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(feature_indices)), consistency_scores)
        plt.xlabel('Feature Index')
        plt.ylabel('Consistency Score (%)')
        plt.title('Feature Consistency Across Degradation Types')
        plt.xticks(range(len(feature_indices)), feature_indices, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_consistency_summary.png'))
        plt.close()
        
        # Create degradation type association heatmap
        unique_degradations = list(set(degradation_associations.values()))
        degradation_matrix = np.zeros((len(unique_degradations), len(feature_indices)))
        
        for i, feature_idx in enumerate(feature_indices):
            deg_type = degradation_associations[feature_idx]
            deg_idx = unique_degradations.index(deg_type)
            degradation_matrix[deg_idx, i] = 1
        
        plt.figure(figsize=(15, 8))
        plt.imshow(degradation_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='Association Strength')
        plt.xlabel('Feature Index')
        plt.ylabel('Degradation Type')
        plt.yticks(range(len(unique_degradations)), unique_degradations)
        plt.title('Feature-Degradation Type Associations')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_degradation_associations.png'))
        plt.close()
        print("Summary visualizations created successfully")
    except Exception as e:
        print(f"Error creating summary visualization: {str(e)}")