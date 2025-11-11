import numpy as np
pi = np.pi
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from utils import Writer, _load_config
import configparser
from scipy import ndimage

CONFIG = _load_config()
DATASET_DEFAULTS = CONFIG.get('dataset_defaults', {})

def _extract_base_params(label):
    return CONFIG['particles'][label]['parameters']

def _generate_size_variations(base_params, n_variations=5):
    variations = []
    scales = [0.4, 0.6, 0.8, 1.2, 1.5]
    for scale in scales[:n_variations]:
        varied = []
        for param in base_params:
            if param[0] == 1:
                varied.append(param)
            else:
                varied.append([int(param[0] * scale)] if len(param) == 1 else [round(param[0] * scale, 1)])
        variations.append(varied)
    return variations

def _generate_param_ranges(base_params, scale_factors=[1.5, 1.0, 0.5]):
    ranged = []
    for param in base_params:
        if param[0] == 1:
            ranged.append([1, 1, 1])
        else:
            base_val = param[0]
            ranged.append([int(base_val * scale_factors[0]), 
                          int(base_val * scale_factors[1]), 
                          int(base_val * scale_factors[2]) if base_val * scale_factors[2] >= 1 else 1])
    return ranged

class Object:
    def __init__(self, x, y, label, parameters, theta=None): 
        self.x = x
        self.y = y
        self.theta = theta 
        self.label = label
        self.parameters = parameters

      
def generateImage(objects, image_w, image_h, snr_range, i_range=[1,1]):
    image = np.zeros([image_w, image_h])
    bboxes = []
    labels = []
    X, Y = np.meshgrid(np.arange(0, image_w), np.arange(0, image_h))
    for obj in objects:
        x = obj.x
        y = obj.y
        #a = np.random.uniform(i_range[0], i_range[1])
        if obj.label == 'Spot':
            i_list, s_list = np.array(obj.parameters)
            i = np.random.uniform(i_range[0], i_range[1]) if i_list[0] == 0 else i_list[0]
            s = np.random.uniform(s_list[0], s_list[1]) if len(s_list) > 1 else s_list[0] 
            image = image + i*np.exp(-((X-x)**2+(Y-y)**2)/(2*s**2))
            bx = 2*s
            by = 2*s
            bboxes.append([[x-bx,y-by],[x+bx,y+by]])
            labels.append(obj.label)
        if obj.label == 'Ring':                
            i_list, r_list, s_list = np.array(obj.parameters)
            i = np.random.uniform(i_range[0], i_range[1]) if i_list[0] == 0 else i_list[0]
            r = np.random.uniform(r_list[0], r_list[1]) if len(r_list) > 1 else r_list[0] 
            s = np.random.uniform(s_list[0], s_list[1]) if len(s_list) > 1 else s_list[0]      
            image = image + i*np.exp(-(np.sqrt((X-x)**2+(Y-y)**2)-r)**2/(2*s**2))
            bx = 2*s + r
            by = 2*s + r
            bboxes.append([[x-bx,y-by],[x+bx,y+by]])
            labels.append(obj.label)
        if obj.label == 'Janus':
            i_list, r_list, s_list = np.array(obj.parameters)
            i = np.random.uniform(i_range[0], i_range[1]) if i_list[0] == 0 else i_list[0]
            r = np.random.uniform(r_list[0], r_list[1]) if len(r_list) > 1 else r_list[0]
            s = np.random.uniform(s_list[0], s_list[1]) if len(s_list) > 1 else s_list[0]
            if obj.theta is None:
                phi = np.random.random()*2*pi
            else:
                phi = obj.theta
            Xr = x + np.cos(phi)*(X-x) - np.sin(phi)*(Y-y)
            Yr = y + np.sin(phi)*(X-x) + np.cos(phi)*(Y-y)
            angle = np.nan_to_num(np.arccos((Xr-x)/np.sqrt(((Xr-x)**2+(Yr-y)**2))))/2
            image = image + np.cos(angle)**2*i*np.exp(-(np.sqrt((X-x)**2+(Y-y)**2)-r)**2/(2*s**2))
            bx = 2*s + r
            by = 2*s + r
            bboxes.append([[x-bx,y-by],[x+bx,y+by], phi/(2*pi)]) # !!!
            labels.append(obj.label)
        if obj.label == 'Ellipse':
            i_list, sx_list, sy_list = np.array(obj.parameters)
            i = np.random.uniform(i_range[0], i_range[1]) if i_list[0] == 0 else i_list[0]
            sx = np.random.uniform(sx_list[0], sx_list[1]) if len(sx_list) > 1 else sx_list[0] 
            sy = np.random.uniform(sy_list[0], sy_list[1]) if len(sy_list) > 1 else sy_list[0]
            if obj.theta is None:
                theta = np.random.uniform(0, 2*pi) 
            else:
                theta = obj.theta
            a = np.cos(theta)**2/(2*sx**2) + np.sin(theta)**2/(2*sy**2)
            b = -np.sin(2*theta)/(4*sx**2) + np.sin(2*theta)/(4*sy**2)
            c = np.sin(theta)**2/(2*sx**2) + np.cos(theta)**2/(2*sy**2)
            image = image + i*np.exp(-(a*(X-x)**2 + 2*b*(X-x)*(Y-y) + c*(Y-y)**2))
            bx = 2*sx # !!!
            by = 2*sy # !!!
            bboxes.append([[x-bx,y-by],[x+bx,y+by], theta/(2*pi)]) # !!!
            labels.append(obj.label)
        if obj.label == 'Rod':
            i_list, l_list, w_list, s_list = np.array(obj.parameters)
            i = np.random.uniform(i_range[0], i_range[1]) if i_list[0] == 0 else i_list[0]
            l = np.random.uniform(l_list[0], l_list[1]) if len(l_list) > 1 else l_list[0] 
            w = np.random.uniform(w_list[0], w_list[1]) if len(w_list) > 1 else w_list[0] 
            s = np.random.uniform(s_list[0], s_list[1]) if len(s_list) > 1 else s_list[0]
            if obj.theta is None:
                theta = np.random.uniform(0, 2*pi) 
            else:
                theta = obj.theta
            im = np.zeros([image_w, image_h])
            im[int(image_w/2-w/2):int(-image_w/2+w/2), int(image_h/2-l/2):int(-image_h/2+l/2)] = 1
            im = ndimage.rotate(im, np.degrees(theta), reshape=False, mode='constant')
            im = ndimage.shift(im, (y-int(image_h/2)+0.5, x-int(image_w/2)+0.5))
            im = ndimage.gaussian_filter(im, s)
            im /= im.max()
            image = image + i*im
            bx = l/2 + 2*s
            by = w/2 + 2*s
            bboxes.append([[x-bx,y-by],[x+bx,y+by], theta/(2*pi)])
            labels.append(obj.label)

    # Set the SNR  
    image = image/image.max()
    noise = np.abs(np.random.randn(image_w, image_h))
    noise = noise/np.var(noise)
    if isinstance(snr_range, list):
        snr = np.random.uniform(snr_range[0], snr_range[1])             
    else:
        snr = snr_range
    image = snr*image + noise                    
    return (bboxes, labels, image, snr)

def getRandom(frames, n_list, image_w, image_h, distance, offset, label_list, parameters_list):
 
    if not isinstance(n_list, list): 
        n_list = [n_list]
    if not isinstance(label_list, list):
        label_list = [label_list]
    if len(n_list) != len(label_list):
        raise ValueError('The lists must have equal length')
    
    objects = []
    for _ in range(frames):
        positions = np.random.random(2)*(image_w - 2*offset) + offset
        for _ in range(np.sum(n_list)-1):
            min_distance = 0
            while min_distance < distance:
                new_pos = np.random.random(2)*(image_w - 2*offset) + offset
                pos = positions.reshape(int(len(positions)/2), 2)
                d = pos - new_pos
                min_distance = np.sqrt(np.sum(d*d, axis=1)).min()
            positions = np.append(positions, new_pos)
        #if isinstance(labels, list):
        repeated_labels = []
        repeated_parameters = []
        for i, n in enumerate(n_list):
            repeated_labels.extend([label_list[i]] * n)
            repeated_parameters.extend([parameters_list[i]] * n)
        
        objects.append([Object(x,y, label, parameters) for (x, y), label, parameters in zip(positions.reshape(np.sum(n_list), 2), 
                                                                                            repeated_labels, 
                                                                                            repeated_parameters)])
        #else:
        #    objects_list.append([Object(x,y, labels) for x, y in positions.reshape(n, 2)])      
    return np.array(objects)


def getTrajectories(T, dt, D_list, scale, n_list, image_w, image_h, label_list, parameters_list):
    
    # error handling
    if not isinstance(D_list, list): 
        D_list = [D_list]
    if not isinstance(n_list, list): 
        n_list = [n_list]
    if not isinstance(label_list, list):
        label_list = [label_list]
    length = len(D_list)
    if not all(len(l) == length for l in [n_list, label_list]):
        raise ValueError('The lists must have equal length')
        
    frames = int(T/dt)       
    objects = []
    positions = np.random.uniform((0, 0), (image_w, image_h), size=(np.sum(n_list), 2))
    for _ in range(frames):
        for n, n_cs, D in zip(n_list, np.cumsum(n_list), D_list):
            positions[n_cs-n:n_cs] += np.sqrt(2*D*dt)*np.random.normal(size=(n,2))/scale
        repeated_labels = []
        repeated_parameters = []
        for i, n in enumerate(n_list):
            repeated_labels.extend([label_list[i]] * n)
            repeated_parameters.extend([parameters_list[i]] * n)
        
        objects.append([Object(x,y, label, parameters) for (x, y), label, parameters in zip(positions, 
                                                                                            repeated_labels, 
                                                                                            repeated_parameters)])
    return np.array(objects)


def exportConfig(file, nimages, label_list, parameters_list, n_list, snr_range, i_range, distance, offset):
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option  # preserve case for letters

    config.add_section('Section1')
    config.set('Section1', 'nimages', str(nimages))
    config.add_section('Section2')
    config.set('Section2', 'label_list', str(label_list))
    config.set('Section2', 'parameters_list', str(parameters_list))
    config.set('Section2', 'n_list', str(n_list))
    config.add_section('Section3')
    config.set('Section3', 'snr_range', str(snr_range))
    config.set('Section3', 'i_range', str(i_range))
    config.set('Section3', 'distance', str(distance))
    config.set('Section3', 'offset', str(offset))

    with open(file, 'w') as configfile:    # save
        config.write(configfile)
    
    
def rotate(origin, point, angle):
    x0, y0 = origin
    x1, y1 = point
    x2 = x0 + np.cos(angle)*(x1 - x0) - np.sin(angle)*(y1 - y0)
    y2 = y0 - np.sin(angle)*(x1 - x0) - np.cos(angle)*(y1 - y0)
    return x2, y2

def generate_same_shape_same_size_dataset(subdir, nimages, image_w, image_h, distance, offset, snr_range, i_range):
    label_list = ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
    parameters = [_extract_base_params(label) for label in label_list]
    n_list = [1, 5]
    
    dataset_name = 'same_shape_same_size'
    dataset_dir = f'{subdir}/{dataset_name}'
    
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    
    i_dir = f'{dataset_dir}/images/'
    if not os.path.exists(i_dir):
        os.mkdir(i_dir)
    a_dir = f'{dataset_dir}/annotations/'
    if not os.path.exists(a_dir):
        os.mkdir(a_dir)
    
    for i in tqdm(range(nimages), desc=f'Generating {dataset_name}'):
        ind = np.random.randint(0, len(label_list))
        current_label_list = [label_list[ind]]
        current_parameters = [parameters[ind]]
        objects = getRandom(1, np.random.randint(n_list[0], n_list[1] + 1), image_w, image_h, distance, offset, current_label_list, current_parameters)[0]
        bboxes, labels, image, snr = generateImage(objects, image_w, image_h, snr_range, i_range)
        
        fname = f'{i_dir}image_{i:04d}.jpg'
        plt.imsave(fname, image, cmap='gray')
        
        writer = Writer(fname, image_w, image_h)
        writer.setSNR(snr)
        for bbox, label in zip(bboxes, labels):
            xmin, ymin = bbox[0]
            xmax, ymax = bbox[1]
            theta = bbox[2] if len(bbox) > 2 else 0
            writer.addObject(label, xmin, ymin, xmax, ymax, theta)
        
        xmlname = f'{a_dir}image_{i:04d}.xml'
        writer.save(xmlname)
    
    exportConfig(f'{dataset_dir}/info.txt', [nimages], label_list, parameters, n_list, snr_range, i_range, distance, offset)

def generate_same_shape_different_size_dataset(subdir, nimages, image_w, image_h, distance, offset, snr_range, i_range):
    label_list = ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
    configs = {label: _generate_size_variations(_extract_base_params(label), n_variations=5) for label in label_list}
    n_list = [x for x in np.random.randint(1, 4, 5)]
  
    dataset_name = 'same_shape_different_size'
    dataset_dir = f'{subdir}/{dataset_name}'
    
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    
    i_dir = f'{dataset_dir}/images/'
    if not os.path.exists(i_dir):
        os.mkdir(i_dir)
    a_dir = f'{dataset_dir}/annotations/'
    if not os.path.exists(a_dir):
        os.mkdir(a_dir)
    
    for i in tqdm(range(nimages), desc=f'Generating {dataset_name}'):
        ind = np.random.randint(0, len(label_list))
        label = label_list[ind]
        parameters_list = configs[label]
        current_label_list = [label] * len(parameters_list)
        objects = getRandom(1, n_list, image_w, image_h, distance, offset, current_label_list, parameters_list)[0]
        bboxes, labels, image, snr = generateImage(objects, image_w, image_h, snr_range, i_range)
        
        fname = f'{i_dir}image_{i:04d}.jpg'
        plt.imsave(fname, image, cmap='gray')
        
        writer = Writer(fname, image_w, image_h)
        writer.setSNR(snr)
        for bbox, label in zip(bboxes, labels):
            xmin, ymin = bbox[0]
            xmax, ymax = bbox[1]
            theta = bbox[2] if len(bbox) > 2 else 0
            writer.addObject(label, xmin, ymin, xmax, ymax, theta)
        
        xmlname = f'{a_dir}image_{i:04d}.xml'
        writer.save(xmlname)
    
    exportConfig(f'{dataset_dir}/info.txt', [nimages], configs.keys(), configs.values(), n_list, snr_range, i_range, distance, offset)

def generate_different_shape_same_size_dataset(subdir, nimages, image_w, image_h, distance, offset, snr_range, i_range):
    label_list = ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
    parameters_list = [_extract_base_params(label) for label in label_list]
    n_list = [x for x in np.random.randint(1, 10, 5)]
    
    dataset_name = 'different_shape_same_size'
    dataset_dir = f'{subdir}/{dataset_name}'
    
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    
    i_dir = f'{dataset_dir}/images/'
    if not os.path.exists(i_dir):
        os.mkdir(i_dir)
    a_dir = f'{dataset_dir}/annotations/'
    if not os.path.exists(a_dir):
        os.mkdir(a_dir)
    
    for i in tqdm(range(nimages), desc=f'Generating {dataset_name}'):
        objects = getRandom(1, n_list, image_w, image_h, distance, offset, label_list, parameters_list)[0]
        bboxes, labels, image, snr = generateImage(objects, image_w, image_h, snr_range, i_range)
        
        fname = f'{i_dir}image_{i:04d}.jpg'
        plt.imsave(fname, image, cmap='gray')
        
        writer = Writer(fname, image_w, image_h)
        writer.setSNR(snr)
        for bbox, label in zip(bboxes, labels):
            xmin, ymin = bbox[0]
            xmax, ymax = bbox[1]
            theta = bbox[2] if len(bbox) > 2 else 0
            writer.addObject(label, xmin, ymin, xmax, ymax, theta)
        
        xmlname = f'{a_dir}image_{i:04d}.xml'
        writer.save(xmlname)
    
    exportConfig(f'{dataset_dir}/info.txt', [nimages], label_list, parameters_list, n_list, snr_range, i_range, distance, offset)

def generate_different_shape_different_size_dataset(subdir, nimages, image_w, image_h, distance, offset, snr_range, i_range):
    label_list = ['Janus', 'Ring', 'Spot', 'Ellipse', 'Rod']
    parameters_list = [_generate_param_ranges(_extract_base_params(label)) for label in label_list]
    n_list = [x for x in np.random.randint(1,10,5)]
    
    dataset_name = 'different_shape_different_size'
    dataset_dir = f'{subdir}/{dataset_name}'
    
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    
    i_dir = f'{dataset_dir}/images/'
    if not os.path.exists(i_dir):
        os.mkdir(i_dir)
    a_dir = f'{dataset_dir}/annotations/'
    if not os.path.exists(a_dir):
        os.mkdir(a_dir)
    
    for i in tqdm(range(nimages), desc=f'Generating {dataset_name}'):
        objects = getRandom(1, n_list, image_w, image_h, distance, offset, label_list, parameters_list)[0]
        bboxes, labels, image, snr = generateImage(objects, image_w, image_h, snr_range, i_range)
        
        fname = f'{i_dir}image_{i:04d}.jpg'
        plt.imsave(fname, image, cmap='gray')
        
        writer = Writer(fname, image_w, image_h)
        writer.setSNR(snr)
        for bbox, label in zip(bboxes, labels):
            xmin, ymin = bbox[0]
            xmax, ymax = bbox[1]
            theta = bbox[2] if len(bbox) > 2 else 0
            writer.addObject(label, xmin, ymin, xmax, ymax, theta)
        
        xmlname = f'{a_dir}image_{i:04d}.xml'
        writer.save(xmlname)
    
    exportConfig(f'{dataset_dir}/info.txt', [nimages], label_list, parameters_list, n_list, snr_range, i_range, distance, offset)

def main():
    """Main function with CLI support"""
    parser = argparse.ArgumentParser(description='Generate testing datasets for LodeSTAR')
    parser.add_argument('--image-width', type=int, default=DATASET_DEFAULTS['image_width'],
                       help=f"Image width in pixels (default: {DATASET_DEFAULTS['image_width']})")
    parser.add_argument('--image-height', type=int, default=DATASET_DEFAULTS['image_height'],
                       help=f"Image height in pixels (default: {DATASET_DEFAULTS['image_height']})")
    parser.add_argument('--snr-min', type=float, default=DATASET_DEFAULTS['snr_min'],
                       help=f"Minimum SNR (default: {DATASET_DEFAULTS['snr_min']})")
    parser.add_argument('--snr-max', type=float, default=DATASET_DEFAULTS['snr_max'],
                       help=f"Maximum SNR (default: {DATASET_DEFAULTS['snr_max']})")
    parser.add_argument('--intensity-min', type=float, default=DATASET_DEFAULTS['intensity_min'],
                       help=f"Minimum intensity (default: {DATASET_DEFAULTS['intensity_min']})")
    parser.add_argument('--intensity-max', type=float, default=DATASET_DEFAULTS['intensity_max'],
                       help=f"Maximum intensity (default: {DATASET_DEFAULTS['intensity_max']})")
    parser.add_argument('--distance', type=float, default=DATASET_DEFAULTS['distance'],
                       help=f"Minimum distance between particles (default: {DATASET_DEFAULTS['distance']})")
    parser.add_argument('--offset', type=float, default=DATASET_DEFAULTS['offset'],
                       help=f"Offset from image edges (default: {DATASET_DEFAULTS['offset']})")
    parser.add_argument('--nimages', type=int, default=DATASET_DEFAULTS['nimages'],
                       help=f"Number of images per dataset (default: {DATASET_DEFAULTS['nimages']})")
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: data/Testing_snr_MIN-MAX)')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       choices=['same_shape_same_size', 'same_shape_different_size', 
                               'different_shape_same_size', 'different_shape_different_size', 'all'],
                       default=['all'],
                       help='Which datasets to generate (default: all)')
    
    args = parser.parse_args()
    
    if args.snr_min > args.snr_max:
        print(f"Error: SNR min ({args.snr_min}) must be less or equal to SNR max ({args.snr_max})")
        return 1
    
    if args.intensity_min >= args.intensity_max:
        print(f"Error: Intensity min ({args.intensity_min}) must be less than intensity max ({args.intensity_max})")
        return 1
    
    snr_range = [args.snr_min, args.snr_max]
    i_range = [args.intensity_min, args.intensity_max]
    
    if args.output_dir is None:
        subdir = f'data/Testing_snr_{args.snr_min}-{args.snr_max}'
    else:
        subdir = args.output_dir
    
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    
    print(f"=== LodeSTAR Dataset Generator ===")
    print(f"Configuration:")
    print(f"  Image size: {args.image_width}x{args.image_height}")
    print(f"  SNR range: [{args.snr_min}, {args.snr_max}]")
    print(f"  Intensity range: [{args.intensity_min}, {args.intensity_max}]")
    print(f"  Distance: {args.distance}")
    print(f"  Offset: {args.offset}")
    print(f"  Images per dataset: {args.nimages}")
    print(f"  Output directory: {subdir}")
    print()
    
    datasets_to_generate = args.datasets
    if 'all' in datasets_to_generate:
        datasets_to_generate = ['same_shape_same_size', 'same_shape_different_size', 
                               'different_shape_same_size', 'different_shape_different_size']
    
    try:
        print("Generating datasets...")
        
        if 'same_shape_same_size' in datasets_to_generate:
            generate_same_shape_same_size_dataset(subdir, args.nimages, args.image_width, args.image_height, 
                                                 args.distance, args.offset, snr_range, i_range)
        
        if 'same_shape_different_size' in datasets_to_generate:
            generate_same_shape_different_size_dataset(subdir, args.nimages, args.image_width, args.image_height, 
                                                      args.distance, args.offset, snr_range, i_range)
        
        if 'different_shape_same_size' in datasets_to_generate:
            generate_different_shape_same_size_dataset(subdir, args.nimages, args.image_width, args.image_height, 
                                                      args.distance, args.offset, snr_range, i_range)
        
        if 'different_shape_different_size' in datasets_to_generate:
            generate_different_shape_different_size_dataset(subdir, args.nimages, args.image_width, args.image_height, 
                                                           args.distance, args.offset, snr_range, i_range)
        
        print(f"\n=== Generation Complete ===")
        print(f"All datasets generated successfully!")
        print(f"Datasets saved in: {subdir}")
        
    except Exception as e:
        print(f"Error during dataset generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())