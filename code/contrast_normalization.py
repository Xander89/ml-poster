#!/usr/bin/env python
import numpy as np 

#luminance of rgb color
def luminance(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def contrast_normalize(colors):
    r, g, b = colors
    luma = luminance(r,g,b)
    min_luma = np.amin(luma, axis = 1)
    max_luma = np.amax(luma, axis = 1)    
    min_luma_s = np.stack((min_luma for i in range(r.shape[1])), axis = 1)
    max_luma_s = np.stack((max_luma for i in range(r.shape[1])), axis = 1)
    
    epsilon = np.zeros(r.shape)
    epsilon.fill(10e-6)
    ra = (r - min_luma_s)/(max_luma_s - min_luma_s + epsilon)
    ga = (g - min_luma_s)/(max_luma_s - min_luma_s + epsilon)
    ba = (b - min_luma_s)/(max_luma_s - min_luma_s + epsilon)
    return (ra,ga,ba),min_luma,max_luma
    
def contrast_normalize_stat(colors):
#    r, g, b = d["r"]["data"], d["g"]["data"], d["b"]["data"]
    r, g, b = colors
    luma = luminance(r,g,b)
    mean = np.mean(luma, axis = 1)
    std = np.std(luma, axis = 1)    
    mean_s = np.stack((mean for i in range(r.shape[1])), axis = 1)
    std_s = np.stack((std for i in range(r.shape[1])), axis = 1)
    
    epsilon = np.zeros(r.shape)
    epsilon.fill(10e-6)
    ra = (r - mean_s)/(std_s + epsilon)
    ga = (g - mean_s)/(std_s + epsilon)
    ba = (b - mean_s)/(std_s + epsilon)

    return (ra,ga,ba),mean,std
    
def contrast_denormalize(colors,min_luma, max_luma):
    r, g, b = colors
    min_luma_s = np.stack((min_luma for i in range(r.shape[1])), axis = 1)
    max_luma_s = np.stack((max_luma for i in range(r.shape[1])), axis = 1)
    epsilon = np.zeros(r.shape)
    epsilon.fill(10e-6)
    rd = r * (max_luma_s - min_luma_s + epsilon) + min_luma_s
    gd = g * (max_luma_s - min_luma_s + epsilon) + min_luma_s
    bd = b * (max_luma_s - min_luma_s + epsilon) + min_luma_s
    return (rd, gd, bd)

def contrast_denormalize_stat(colors,mean, std):
    r, g, b = colors
    mean_s = np.stack((mean for i in range(r.shape[1])), axis = 1)
    std_s = np.stack((std for i in range(r.shape[1])), axis = 1)
    epsilon = np.zeros(r.shape)
    epsilon.fill(10e-6)
    rd = r * (std_s + epsilon) + mean_s
    gd = g * (std_s + epsilon) + mean_s
    bd = b * (std_s + epsilon) + mean_s
    return (rd, gd, bd)
