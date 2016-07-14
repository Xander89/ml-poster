# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:50:02 2016

@author: aluo
"""


import os
import sys
import argparse
import pickle 
import errno
import inspect

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

if __name__ == '__main__':

    path = get_script_dir()
    print(path)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--training', required=False, default="training", dest="training_folder", help='Folder containing training data.')
    
    args = parser.parse_args(sys.argv[1:])
    training_files = []
    print("Scanning for files...")
    for root, dirs, files in os.walk(args.training_folder):
        for name in files:
            if name.endswith((".dat")):
                full_path = os.path.join(path, os.path.join(root, name))             
                if(os.path.isfile(full_path)) :
                    d = unpickle(full_path)
                    ff = open(full_path, "wb")
                    pickle.dump(d, ff, protocol=pickle.HIGHEST_PROTOCOL)
                    ff.close()
                    
        