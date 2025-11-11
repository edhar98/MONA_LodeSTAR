import matplotlib.pyplot as plt
import os
import argparse
from utils import _load_config
from datetime import datetime
from image_generator import generateImage, Object

# Load configuration
CONFIG = _load_config('samples.yaml')
PARTICLE_CONFIGS = CONFIG['particles']
DEFAULTS = CONFIG['defaults']

def generate_single_particle_sample(particle_type, image_size=None, snr_range=None, i_range=None, force_overwrite=None):
    """Generate a single particle sample image"""
    
    image_size = image_size if image_size is not None else DEFAULTS['image_size']
    snr_range = snr_range if snr_range is not None else DEFAULTS['snr']
    i_range = i_range if i_range is not None else [DEFAULTS['intensity_min'], DEFAULTS['intensity_max']]
    force_overwrite = force_overwrite if force_overwrite is not None else DEFAULTS['force_overwrite']
    
    if particle_type not in PARTICLE_CONFIGS:
        raise ValueError(f"Unknown particle type: {particle_type}. Available types: {list(PARTICLE_CONFIGS.keys())}")
    
    image_w = image_size
    image_h = image_size
    
    # Create Samples directory structure
    samples_dir = 'data/Samples'
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    print(f"Generating {particle_type} sample image...")
    print(f"Image size: {image_w}x{image_h}")
    print(f"SNR range: {snr_range}")
    print(f"Intensity range: {i_range}")
    
    config = PARTICLE_CONFIGS[particle_type]
    
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
    
    file_info = {
        'particle_type': particle_type,
        'sample_path': sample_path,
        'metadata_path': metadata_path,
        'snr': snr,
        'parameters': config['parameters']
    }
    
    print(f"  Saved: {sample_path}")
    print(f"  SNR: {snr:.2f}")
    print(f"  Parameters: {config['parameters']}")
    
    return file_info

def generate_sample_images(image_size=None, snr_range=None, i_range=None, force_overwrite=None):
    """Generate clean sample images for each particle type"""
    
    image_size = image_size if image_size is not None else DEFAULTS['image_size']
    snr_range = snr_range if snr_range is not None else DEFAULTS['snr']
    i_range = i_range if i_range is not None else [DEFAULTS['intensity_min'], DEFAULTS['intensity_max']]
    force_overwrite = force_overwrite if force_overwrite is not None else DEFAULTS['force_overwrite']
    
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

def visualize_single_particle(particle_type, samples_dir='data/Samples'):
    """Visualize a single particle sample"""
    
    sample_path = os.path.join(samples_dir, particle_type, f'{particle_type}.jpg')
    
    if os.path.exists(sample_path):
        plt.figure(figsize=(8, 6))
        img = plt.imread(sample_path)
        plt.imshow(img, cmap='gray')
        plt.title(f'{particle_type} Sample ({img.shape[0]}x{img.shape[1]} pixels)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Sample not found: {sample_path}")

def visualize_samples(samples_dir='data/Samples', particle_types=None):
    """Visualize all generated samples or specific particle types"""
    
    if particle_types is None:
        particle_types = ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
    
    fig, axes = plt.subplots(1, len(particle_types), figsize=(4*len(particle_types), 4))
    
    # Handle single particle case
    if len(particle_types) == 1:
        axes = [axes]
    
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
    parser.add_argument('--image-size', type=int, default=DEFAULTS['image_size'], 
                       help=f"Image size (width=height) in pixels (default: {DEFAULTS['image_size']})")
    parser.add_argument('--snr', type=float, default=DEFAULTS['snr'],
                       help=f"Signal-to-noise ratio (default: {DEFAULTS['snr']})")
    parser.add_argument('--intensity-min', type=float, default=DEFAULTS['intensity_min'],
                       help=f"Minimum intensity (default: {DEFAULTS['intensity_min']})")
    parser.add_argument('--intensity-max', type=float, default=DEFAULTS['intensity_max'],
                       help=f"Maximum intensity (default: {DEFAULTS['intensity_max']})")
    parser.add_argument('--force-overwrite', action='store_true', default=DEFAULTS['force_overwrite'],
                       help=f"Force overwrite existing files (default: {DEFAULTS['force_overwrite']})")
    parser.add_argument('--visualize', action='store_true', default=DEFAULTS['visualize'],
                       help=f"Visualize generated samples after generation (default: {DEFAULTS['visualize']})")
    parser.add_argument('--particle-type', type=str, choices=['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod', 'all'],
                       default=DEFAULTS['particle_type'], help=f"Generate specific particle type or all (default: {DEFAULTS['particle_type']})")
    
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
            single_file = generate_single_particle_sample(
                particle_type=args.particle_type,
                image_size=args.image_size,
                snr_range=args.snr,
                i_range=i_range,
                force_overwrite=args.force_overwrite
            )
            generated_files = [single_file]  # Convert to list for consistency
        
        # Visualize if requested
        if args.visualize:
            print("\nVisualizing generated samples...")
            if args.particle_type == 'all':
                visualize_samples()
            else:
                visualize_single_particle(args.particle_type)
        
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