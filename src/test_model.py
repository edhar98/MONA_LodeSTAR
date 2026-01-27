import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from deeplay import LodeSTAR
import deeptrack.deeplay as dl
import deeptrack as dt
import utils
from image_generator import generateImage, Object


def load_trained_model(checkpoint_path):
    """Load trained LodeSTAR model"""
    
    # Load configuration
    config = utils.load_yaml('config.yaml')
    
    # Create LodeSTAR model
    lodestar = dl.LodeSTAR(
        n_transforms=config['n_transforms'], 
        optimizer=dl.Adam(lr=config['lr'])
    ).build()
    
    # Load trained weights
    if os.path.exists(checkpoint_path):
        lodestar.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return None
    
    lodestar.eval()
    return lodestar


def test_on_generated_images(model, config):
    """Test model on generated images with different particle types"""
    
    particle_configs = {
        'Janus': {'parameters': [[1], [8], [2]], 'description': 'Janus particle'},
        'Ring': {'parameters': [[1], [8], [2]], 'description': 'Ring particle'},
        'Spot': {'parameters': [[1], [3]], 'description': 'Spot particle'},
        'Ellipse': {'parameters': [[1], [8], [4]], 'description': 'Elliptical particle'},
        'Rod': {'parameters': [[1], [12], [4], [1.5]], 'description': 'Rod particle'}
    }
    
    results = {}
    
    for particle_type, config_particle in particle_configs.items():
        print(f"\nTesting {particle_type}...")
        
        # Generate test image
        particle = Object(
            x=64, y=64,  # Center of 128x128 image
            label=particle_type,
            parameters=config_particle['parameters']
        )
        
        bboxes, labels, image, snr = generateImage(
            [particle], 
            128, 128,  # Image size
            15,  # SNR
            [0.8, 1.0]  # Intensity range
        )
        
        # Prepare image for model
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        
        # Run inference
        with torch.no_grad():
            predictions = model(image_tensor)
        
        # Store results
        results[particle_type] = {
            'image': image,
            'predictions': predictions,
            'bboxes': bboxes,
            'snr': snr
        }
        
        print(f"  SNR: {snr:.2f}")
        print(f"  Predictions shape: {predictions.shape}")
    
    return results


def visualize_results(results):
    """Visualize test results"""
    
    n_particles = len(results)
    fig, axes = plt.subplots(2, n_particles, figsize=(4*n_particles, 8))
    
    if n_particles == 1:
        axes = axes.reshape(2, 1)
    
    for i, (particle_type, result) in enumerate(results.items()):
        # Original image
        axes[0, i].imshow(result['image'], cmap='gray')
        axes[0, i].set_title(f'{particle_type}\nSNR: {result["snr"]:.1f}')
        axes[0, i].axis('off')
        
        # Model predictions
        pred_image = result['predictions'][0, 0].cpu().numpy()
        axes[1, i].imshow(pred_image, cmap='hot')
        axes[1, i].set_title(f'{particle_type}\nPredictions')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_on_real_samples(model, config):
    """Test model on real sample images"""
    
    particle_types = ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
    samples_dir = os.path.join(config['data_dir'], 'Samples')
    
    results = {}
    
    for particle_type in particle_types:
        sample_path = os.path.join(samples_dir, particle_type, f'{particle_type}.jpg')
        
        if os.path.exists(sample_path):
            print(f"\nTesting on {particle_type} sample...")
            
            # Load sample image
            image = np.array(dt.LoadImage(sample_path).resolve()).astype(np.float32)
            
            # Prepare for model
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
            
            # Run inference
            with torch.no_grad():
                predictions = model(image_tensor)
            
            results[particle_type] = {
                'image': image,
                'predictions': predictions
            }
            
            print(f"  Image shape: {image.shape}")
            print(f"  Predictions shape: {predictions.shape}")
        else:
            print(f"Warning: {particle_type} sample not found")
    
    return results


def main():
    """Main testing function"""
    
    # Load configuration
    config = utils.load_yaml('src/config.yaml')
    
    # Find latest checkpoint
    checkpoint_dir = 'lightning_logs'
    if os.path.exists(checkpoint_dir):
        # Find the most recent checkpoint
        checkpoint_files = []
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith('.ckpt'):
                    checkpoint_files.append(os.path.join(root, file))
        
        if checkpoint_files:
            # Sort by modification time
            checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_checkpoint = checkpoint_files[0]
            print(f"Found checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint files found")
            return
    else:
        print("Checkpoint directory not found")
        return
    
    # Load model
    model = load_trained_model(latest_checkpoint)
    if model is None:
        return
    
    print("\n=== Testing on Generated Images ===")
    generated_results = test_on_generated_images(model, config)
    visualize_results(generated_results)
    
    print("\n=== Testing on Sample Images ===")
    sample_results = test_on_real_samples(model, config)
    visualize_results(sample_results)
    
    print("\nTesting complete!")


if __name__ == '__main__':
    main() 