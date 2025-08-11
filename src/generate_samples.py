import numpy as np
import matplotlib.pyplot as plt
import os
from image_generator import generateImage, Object

def generate_sample_images():
    """Generate clean sample images for each particle type"""
    
    # Sample image parameters
    image_w = 50  # Smaller size for samples
    image_h = 50
    snr_range = 10  # High SNR for clean samples
    i_range = [0.8, 1.0]  # Good intensity
    
    # Create Samples directory structure
    samples_dir = 'data/Samples'
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    # Particle configurations for clean samples
    particle_configs = {
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
    
    print("Generating sample images for LodeSTAR training...")
    
    for particle_type, config in particle_configs.items():
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
        
        # Save sample image
        sample_path = os.path.join(particle_dir, f'{particle_type}.jpg')
        plt.imsave(sample_path, image, cmap='gray')
        
        # Save metadata
        metadata_path = os.path.join(particle_dir, f'{particle_type}_info.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Particle Type: {particle_type}\n")
            f.write(f"Description: {config['description']}\n")
            f.write(f"Parameters: {config['parameters']}\n")
            f.write(f"Image Size: {image_w}x{image_h}\n")
            f.write(f"SNR: {snr:.2f}\n")
            f.write(f"Bounding Box: {bboxes[0]}\n")
        
        print(f"  Saved: {sample_path}")
        print(f"  SNR: {snr:.2f}")
        print(f"  Parameters: {config['parameters']}")
    
    print("\nSample generation complete!")
    print("Generated samples:")
    for particle_type in particle_configs.keys():
        print(f"  - {particle_type}: data/Samples/{particle_type}/{particle_type}.jpg")

def visualize_samples():
    """Visualize all generated samples"""
    
    samples_dir = 'data/Samples'
    particle_types = ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Sample Images for LodeSTAR Training', fontsize=16)
    
    for i, particle_type in enumerate(particle_types):
        sample_path = os.path.join(samples_dir, particle_type, f'{particle_type}.jpg')
        
        if os.path.exists(sample_path):
            image = plt.imread(sample_path)
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'{particle_type}')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'{particle_type}\nNot Found', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/Samples/sample_overview.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    generate_sample_images()
    visualize_samples() 