# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 10:52:47 2016

@author: alcor
"""

#!/usr/bin/env python
import inspect
import os
import sys
import argparse
from numpy import fromfile
import numpy as np 
import scipy.misc
import pickle

import errno
from contrast_normalization import contrast_normalize, contrast_denormalize

rgb_names = ["r", "g", "b"]

def unpickle(file):
    fo = open(file, 'rb')
    d = pickle.load(fo)
    fo.close()
    return d

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
            
def get_script_dir(follow_symlinks=True):
    if getattr(sys, 'frozen', False): # py2exe, PyInstaller, cx_Freeze
        path = os.path.abspath(sys.executable)
    else:
        path = inspect.getabsfile(get_script_dir)
    if follow_symlinks:
        path = os.path.realpath(path)
    return os.path.dirname(path)

def get_script_complete_path(follow_symlinks = True):
    return get_script_dir(follow_symlinks) + "\\" + sys.argv[0]

def recombine_image(d, output_name = "test.png"):
    patch_size = d["patch_size"]
    pad_size = d["pad_size"]
    image_size = d["image_size"]
    step = patch_size - pad_size
    n_patches = image_size // step
    data = np.zeros([image_size[0], image_size[1],3])
    data_weights = np.zeros([image_size[0], image_size[1],3])
    ones = np.zeros([image_size[0], image_size[1],3])
    ones.fill(1.)
    patch_weight = np.zeros(patch_size)
    patch_weight.fill(1.)
    
    if "normalized" in d.keys() and d["normalized"] is True:
        min_l = d["min_l"]
        max_l = d["max_l"]
        colors = contrast_denormalize((d[color]["data"] for color in rgb_names), min_l, max_l)
        for i in range(3):
            d[rgb_names[i]]["data"] = colors[i]
            
    for color_i in range(3):
        color = d[rgb_names[color_i]]["data"]
        for k in range(n_patches[0]*n_patches[1]):
            y = k % n_patches[1]
            x = (k - y) // n_patches[1]
            if(k < len(color)):
                patch = color[k]
                patch = np.reshape(patch, patch_size)
                initial_pixel = np.array([x,y])*step
                final_pixel = initial_pixel + patch_size
                data[initial_pixel[0]:final_pixel[0],:, color_i][:,initial_pixel[1]:final_pixel[1]] += patch
                data_weights[initial_pixel[0]:final_pixel[0],:, color_i][:,initial_pixel[1]:final_pixel[1]] += patch_weight
    data_weights = np.maximum(data_weights, ones)
    data = data / data_weights
    

    scipy.misc.toimage(data, cmin=0.0, cmax=1.0, channel_axis=2).save(output_name)
    return data
 
 
def extract_patches(colors, dimensions, pad_size, patch_size, fi, normalize_contrast = False, output_images = False):
    step = patch_size - pad_size
    n_patches = dimensions // step
    d = {}
    for i in range(3):
        color = np.flipud(colors[i])
        name = rgb_names[i]
        file_base = fi + "_" + name 
        patches = np.zeros((n_patches[0]*n_patches[1], patch_size[0]*patch_size[1]))
        for j in range(n_patches[0]):
            for k in range(n_patches[1]):
                initial_pixel = np.array([j,k])*step
                final_pixel = initial_pixel + patch_size
                patch = color[initial_pixel[0]:final_pixel[0],:][:,initial_pixel[1]:final_pixel[1]]
                if output_images:
                    scipy.misc.toimage(patch, cmin=0.0, cmax=1.0).save(file_base+ "_" + str(j) + "_" + str(k) + ".png")
                patch = np.reshape(patch, patch.size)
                patches[j * n_patches[1] + k] = patch
        d[name] = {"data" : patches} 
    d["normalized"] = normalize_contrast
    if normalize_contrast:
        colors, m, a = contrast_normalize((d[color]["data"] for color in rgb_names))
        for i in range(3):
            d[rgb_names[i]]["data"] = colors[i]
        d["min_l"] = m
        d["max_l"] = a
        
    d["patch_size"] = patch_size
    d["pad_size"] = pad_size
    d["image_size"] = dimensions
    return d

#gives a matrix random matrix of choices for the patches
def get_random_patches_selection(d, percentage = 0.1):
    data_size = d["r"]["data"].shape[0];
    return np.random.choice(data_size, int(data_size * percentage), replace=False)

def extract_random_patches_dict(d, chosen_patches):
    new_d = {}
    new_d["patch_size"] = d["patch_size"]
    new_d["pad_size"] = d["pad_size"]
    new_d["image_size"] = d["image_size"]
    
    for i in range(3):
        channel = rgb_names[i]
        data = d[channel]["data"]
        selected_patches = data[chosen_patches,:] 
        new_d[channel] = {"data":selected_patches}
    return new_d        


def run(): 
    path = get_script_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', required=False, default=".", dest="input_folder", help='The input folder where to find the images.')
    parser.add_argument('-o','--out', required=False, default="image_patch_data", dest="output_folder", help='The output file.')
    parser.add_argument('-n','--patch-size', required=False, nargs=2, default="16 16", dest="patch_size", help='The patch size for the image.')
    parser.add_argument('-p','--padding-size', required=False, nargs=2, default="2 2", dest="pad_size", help='The overlapping size for each patch.')
    parser.add_argument('--do-not-normalize', required=False, action="store_false",dest="normalize", help="Do not contrast normalize input data.")    
    
    args = parser.parse_args(sys.argv[1:])
    
    tuples = []
    print("Scanning for files...")
    for root, dirs, files in os.walk(args.input_folder):
        for name in files:
            if name.endswith((".txt")):
                full_path = os.path.join(root, name)
                raw_name = os.path.join(root, name[:-4] + ".raw")
                if(os.path.isfile(raw_name)) :
                    print("Found raw file: " + name + " " +  name[:-4] + ".txt")    
                    tuples.append((full_path, raw_name, name[:-4]))
        
    output = args.output_folder
    print("Starting processing of files...")
    
    
    patch_size = np.array([int(s) for s in args.patch_size.split()])
    pad_size = np.array([int(s) for s in args.pad_size.split()])
    make_sure_path_exists(output)
    
    for t in tuples:
        path = output
        file_base = path + "/" + t[2] 
        print("Processing " + t[0] +" in folder "+ path + "... (normalize = " + str(args.normalize) + ")")
        f = open(t[0])
        lines = [line.rstrip('\n') for line in f]    
        frame = int(lines[0])
        width, height = [int(i) for i in lines[1].split(' ')]      
    
        print("Found file with w=" + str(width) + ", h=" + str(height) + ", frame=" + str(frame))
        f.close()
        raw_f = open(t[1])
        data = fromfile(raw_f, np.float32, -1)
        raw_f.close()
        colors = [data[range(i,data.size,3)] for i in range(3)]
        colors = [np.reshape(r, (width, height)) for r in colors]
        d = extract_patches(colors, np.array([width, height]), pad_size, patch_size, file_base, False)
        ff = open(path + "/" + t[2] + ".dat", "wb")
        pickle.dump(d, ff)
        ff.close()
        
if __name__ == '__main__':
    run()
