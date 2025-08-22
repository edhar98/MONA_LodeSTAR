import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from image_generator import generateImage, Object

# Particle configurations for clean samples (moved outside function)
PARTICLE_CONFIGS = {
    'Janus': {
        'parameters': [[1], [8], [2]],  # intensity, radius, sigma
        'description': 'Janus particle with clear hemispherical structure'
    },
    'Ring': {
        'parameters': [[1], [8], [2]],  # intensity, radius, sigma
        'description': 'Ring particle with clear circular structure'
    },
    'Spot': {
        'parameters': [[1], [3]],  # intensity, sigma
        'description': 'Spot particle with Gaussian profile'
    },
    'Ellipse': {
        'parameters': [[1], [8], [4]],  # intensity, sx, sy
        'description': 'Elliptical particle with clear shape'
    },
    'Rod': {
        'parameters': [[1], [12], [4], [1.5]],  # intensity, length, width, sigma
        'description': 'Rod particle with clear rectangular shape'
    }
}

def generate_sample_images(image_size=50, snr_range=10, i_range=[0.8, 1.0], force_overwrite=False):
    """Generate clean sample images for each particle type"""
    
    # Sample image parameters
    image_w = image_size
    image_h = image_size
    
    # Create Samples directory structure
    samples_dir = 'data/Samples'
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    print(f"Generating sample images for LodeSTAR training...")
    print(f"Image size: {image_w}x{image_h}")
    print(f"SNR range: {snr_range}")
    print(f"Intensity range: {i_range}")
    
    generated_files = []
    
    for particle_type, config in PARTICLE_CONFIGS.items():
        print(f"Generating {particle_type} sample...")
        
        # Create particle object at center of image
        particle = Object(
            x=image_w//2, 
            y=image_h//2, 
            label=particle_type, 
            parameters=config['parameters']
        )
        
        # Generate clean image
        bboxes, labels, image, snr = generateImage(
            [particle], 
            image_w, 
            image_h, 
            snr_range, 
            i_range
        )
        
        # Create particle-specific directory
        particle_dir = os.path.join(samples_dir, particle_type)
        if not os.path.exists(particle_dir):
            os.makedirs(particle_dir)
        
        # Generate unique filenames to avoid overwrites
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if files exist and handle overwrite policy
        sample_path = os.path.join(particle_dir, f'{particle_type}.jpg')
        metadata_path = os.path.join(particle_dir, f'{particle_type}_info.txt')
        
        if not force_overwrite:
            if os.path.exists(sample_path) or os.path.exists(metadata_path):
                # Create backup with timestamp
                backup_dir = os.path.join(particle_dir, 'backups')
                os.makedirs(backup_dir, exist_ok=True)
                
                # Move existing files to backup
                if os.path.exists(sample_path):
                    backup_sample = os.path.join(backup_dir, f'{particle_type}_{timestamp}_backup.jpg')
                    os.rename(sample_path, backup_sample)
                    print(f"  Moved existing sample to: {backup_sample}")
                
                if os.path.exists(metadata_path):
                    backup_metadata = os.path.join(backup_dir, f'{particle_type}_{timestamp}_backup_info.txt')
                    os.rename(metadata_path, backup_metadata)
                    print(f"  Moved existing metadata to: {backup_metadata}")
        
        # Save sample image
        plt.imsave(sample_path, image, cmap='gray')
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            f.write(f"Particle Type: {particle_type}\n")
            f.write(f"Description: {config['description']}\n")
            f.write(f"Parameters: {config['parameters']}\n")
            f.write(f"Image Size: {image_w}x{image_h}\n")
            f.write(f"SNR: {snr:.2f}\n")
            f.write(f"Bounding Box: {bboxes[0]}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image Size: {image_w}x{image_h}\n")
            f.write(f"SNR: {snr:.2f}\n")
        
        generated_files.append({
            'particle_type': particle_type,
            'sample_path': sample_path,
            'metadata_path': metadata_path,
            'snr': snr,
            'parameters': config['parameters']
        })
        
        print(f"  Saved: {sample_path}")
        print(f"  SNR: {snr:.2f}")
        print(f"  Parameters: {config['parameters']}")
    
    print("\nSample generation complete!")
    print("Generated samples:")
    for file_info in generated_files:
        print(f"  - {file_info['particle_type']}: {file_info['sample_path']}")
    
    return generated_files

def visualize_samples(samples_dir='data/Samples'):
    """Visualize all generated samples"""
    
    particle_types = ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Sample Images for LodeSTAR Training', fontsize=16)
    
    for i, particle_type in enumerate(particle_types):
        sample_path = os.path.join(samples_dir, particle_type, f'{particle_type}.jpg')
        
        if os.path.exists(sample_path):
            img = plt.imread(sample_path)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'{particle_type}')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'{particle_type}\nNot Found', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{particle_type} (Missing)')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/Samples/sample_overview.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function with CLI support"""
    parser = argparse.ArgumentParser(description='Generate sample images for LodeSTAR training')
    parser.add_argument('--image-size', type=int, default=50, 
                       help='Image size (width=height) in pixels (default: 50)')
    parser.add_argument('--snr', type=float, default=10.0,
                       help='Signal-to-noise ratio (default: 10.0)')
    parser.add_argument('--intensity-min', type=float, default=0.8,
                       help='Minimum intensity (default: 0.8)')
    parser.add_argument('--intensity-max', type=float, default=1.0,
                       help='Maximum intensity (default: 1.0)')
    parser.add_argument('--force-overwrite', action='store_true',
                       help='Force overwrite existing files (default: False)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize generated samples after generation (default: False)')
    parser.add_argument('--particle-type', type=str, choices=['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod', 'all'],
                       default='all', help='Generate specific particle type or all (default: all)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.image_size < 32:
        print("Warning: Image size < 32 may be too small for effective training")
    elif args.image_size > 512:
        print("Warning: Image size > 512 may be computationally expensive")
    
    if args.snr < 1.0:
        print("Warning: SNR < 1.0 may result in very noisy images")
    
    # Set intensity range
    i_range = [args.intensity_min, args.intensity_max]
    
    print(f"=== LodeSTAR Sample Generator ===")
    print(f"Configuration:")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  SNR: {args.snr}")
    print(f"  Intensity range: {i_range}")
    print(f"  Force overwrite: {args.force_overwrite}")
    print(f"  Particle type: {args.particle_type}")
    print()
    
    # Generate samples
    try:
        if args.particle_type == 'all':
            generated_files = generate_sample_images(
                image_size=args.image_size,
                snr_range=args.snr,
                i_range=i_range,
                force_overwrite=args.force_overwrite
            )
        else:
            # Generate single particle type
            print(f"Generating only {args.particle_type} sample...")
            # This would require modifying the function to handle single particle types
            print("Single particle generation not yet implemented - generating all types")
            generated_files = generate_sample_images(
                image_size=args.image_size,
                snr_range=args.snr,
                i_range=i_range,
                force_overwrite=args.force_overwrite
            )
        
        # Visualize if requested
        if args.visualize:
            print("\nVisualizing generated samples...")
            visualize_samples()
        
        print(f"\n=== Generation Complete ===")
        print(f"Generated {len(generated_files)} sample files")
        print(f"Use --visualize flag to view the samples")
        
    except Exception as e:
        print(f"Error during sample generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 