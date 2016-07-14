# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:50:02 2016

@author: aluo
"""


import os
import sys
import argparse
from generate_patches import get_script_dir, make_sure_path_exists
from SdADenoising import loadTrainedData as loadTrainedDataSdA
from ImageDenoising import loadTrainedData as loadTrainedDatadA, loadDataset, saveImage
from SdADenoising import filterImagesSdA

if __name__ == '__main__':

    path = get_script_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', required=False, default="./image_patch_data/sponzat_0_5.dat", dest="image", help='The input image file.')
    parser.add_argument('-o','--out', required=False, default="filtered_images", dest="output_folder", help='The output file.')
    parser.add_argument('-t','--training', required=False, default="training", dest="training_folder", help='Folder containing training data.')
    
    args = parser.parse_args(sys.argv[1:])
   
    if os.path.splitext(args.image)[1]!=".dat":
        print("Not Supported Input Image")
        sys.exit(1)
    dataset_name = os.path.splitext(os.path.basename(args.image))[0]
    patches, datasets = loadDataset(dataset_name)
    training_files = []
    print("Scanning for files...")
    for root, dirs, files in os.walk(args.training_folder):
        for name in files:
            if name.endswith((".dat")):
                full_path = os.path.join(path, os.path.join(root, name))             
                if(os.path.isfile(full_path)) :
                    training_files.append(full_path)
        
    output = os.path.join(path, args.output_folder)
    make_sure_path_exists(output)
    print("Starting processing of files...")
    
    for training_set_file in training_files:
        training_set_name = os.path.basename(training_set_file)
        index_type = training_set_name.find("SdA")
        if index_type == -1:
            index_type = training_set_name.find("dA")
            if index_type == -1:
                print("Unsupported data " + training_set_name)
                continue
            else:
                isSdA = False
        else:
            isSdA = True
            
        if isSdA:
            print("Filtering on " + training_set_name + " ...")
            sda = loadTrainedDataSdA(training_set_file)
            d = filterImagesSdA(datasets, sda)
            saveImage(d, dataset_name + "_" + training_set_name[index_type:], output)
            
            