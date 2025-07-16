# YOLOv2.1
import numpy as np
pi = np.pi
import matplotlib.pyplot as plt

import os
import matplotlib.patches as patches

from tqdm import tqdm # Progress bar

from xml_writer import Writer # Writes XML files in the Pascal VOC format 
import configparser
from skimage.draw import rectangle
from scipy import ndimage

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
            bboxes.append(np.array([[x-bx,y-by],[x+bx,y+by]]))
            labels.append(obj.label)
        if obj.label == 'Ring':                
            i_list, r_list, s_list = np.array(obj.parameters)
            i = np.random.uniform(i_range[0], i_range[1]) if i_list[0] == 0 else i_list[0]
            r = np.random.uniform(r_list[0], r_list[1]) if len(r_list) > 1 else r_list[0] 
            s = np.random.uniform(s_list[0], s_list[1]) if len(s_list) > 1 else s_list[0]      
            image = image + i*np.exp(-(np.sqrt((X-x)**2+(Y-y)**2)-r)**2/(2*s**2))
            bx = 2*s + r
            by = 2*s + r
            bboxes.append(np.array([[x-bx,y-by],[x+bx,y+by]]))
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
            bboxes.append(np.array([[x-bx,y-by],[x+bx,y+by], theta/(2*pi)])) # !!!
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
            bboxes.append(np.array([[x-bx,y-by],[x+bx,y+by], theta/(2*pi)]))
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
    return (bboxes, labels, image) 

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
        objects.append([Object(x,y, label, parameters) for (x, y), label, parameters in zip(positions.reshape(np.sum(n_list), 2), 
                                                                                            np.repeat(label_list, n_list).tolist(), 
                                                                                            np.repeat(np.array(parameters_list), n_list, axis=0).tolist())])
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
        objects.append([Object(x,y, label, parameters) for (x, y), label, parameters in zip(positions, 
                                                                                            np.repeat(label_list, n_list).tolist(), 
                                                                                            np.repeat(np.array(parameters_list), n_list, axis=0).tolist())])
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

def main():
    image_w = 416
    image_h = 416
    n_list = [1, 15]
    label_list = ['Janus', 'Ring', 'Rod', 'Spot', 'Ellipse'] 
    color_list = ['lime']
    parameters_list= [[[1], [15], [3]]] 
    snr_range = [1, 30]
    i_range = [0.1, 1]
    distance = 15
    offset = 15

    objects = getRandom(1, np.random.randint(n_list[0], n_list[1] + 1), image_w, image_h, distance, offset, label_list, parameters_list)[0]

    bboxes, labels, image = generateImage(objects, image_w, image_h, snr_range, i_range)

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.imshow(image, cmap='gray')
    for bbox, label in zip(bboxes, labels):
        xmin, ymin = bbox[0]
        xmax, ymax = bbox[1]
        w, h = xmax - xmin, ymax - ymin 
        x0, y0 =  xmin + w/2, ymin + h/2
        angle = 2*pi*bbox[2]
        ax.add_patch(patches.Polygon([rotate((x0, y0), (xmin, ymin), angle), rotate((x0, y0), (xmax, ymin), angle), rotate((x0, y0), (xmax, ymax), angle), rotate((x0, y0), (xmin, ymax), angle)], linewidth=1, edgecolor=color_list[0], facecolor='none'))
        ax.plot([x0, x0 + 0.5*w*np.cos(angle)], [y0, y0 - 0.5*w*np.sin(angle)], '-', c=color_list[0])
        
    ax.set_xlim(0, image_w)
    ax.set_ylim(image_h, 0)
    plt.show()


    subdir = '../data/Testing' 
    nimages = [100]
    folders = label_list

    if not os.path.exists(subdir):
        os.mkdir(subdir)
        
    for i, prefix in enumerate(folders):
        
        i_dir = subdir + '/' + prefix + '_images/'
        if not os.path.exists(i_dir):
            os.mkdir(i_dir)
        a_dir = subdir + '/' + prefix + '_annotations/'
        if not os.path.exists(a_dir):
            os.mkdir(a_dir)
        
        for i in tqdm(range(nimages[i])):    
            objects = ig.getRandom(1, np.random.randint(n_list[0], n_list[1] + 1), image_w, image_h, distance, offset, label_list, parameters_list)[0]
            bboxes, labels, image = ig.generateImage(objects, image_w, image_h, snr_range, i_range) 
            fname = i_dir + 'image_{:04d}.jpg'.format(i,2)
            plt.imsave(fname, image, cmap='gray')
            writer = Writer(fname, image_w, image_h)
            for bbox, label in zip(bboxes, labels):
                xmin, ymin = bbox[0]
                xmax, ymax = bbox[1]
                theta = bbox[2]
                writer.addObject(label, xmin, ymin, xmax, ymax, theta)
            xmlname = a_dir + 'image_{:04d}.xml'.format(i,2)    
            writer.save(xmlname)
            
    exportConfig(subdir + 'info.txt', nimages, label_list, parameters_list, n_list, snr_range, i_range, distance, offset)

if __name__ == '__main__':
    main()