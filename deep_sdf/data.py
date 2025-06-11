#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

#import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import pytorch_volumetric as pv

import deep_sdf.workspace as ws


def load_data_from_file(filename:str, resolution:float, query_range):
    """
    Load SDF samples from a file and subsample them.

    Args:
        filename (str): Path to the .obj file
        subsample (int): Number of samples to subsample from the file.

    Returns:
        tensor: A tensor containing the subsampled SDF points and their coordinates. 
    """

    obj = pv.MeshObjectFactory(filename)
    sdf = pv.MeshSDF(obj)

    coords, pts = pv.get_coordinates_and_points_in_grid(resolution, query_range, device='cpu')
    sdf_val, sdf_grad = sdf(pts)

    df_pts = pts
    df_sdf = sdf_val.reshape(sdf_val.shape[0], 1)

    to_ret = torch.cat((df_pts, df_sdf), axis=1)

    return to_ret

class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        """
        Args:
            data_source (str): Path to the data source directory.
            subsample (int): Number of samples to subsample from each SDF file.
            load_ram (bool): Whether to load data into RAM.
            print_filename (bool): Whether to print the filename being loaded.
            num_files (int): Maximum number of files to load.
        """
        self.subsample = subsample

        self.query_range = np.array([
            [0, 1],
            [0, 1],
            [0, 0],  
        ])

        self.resolution = 1 / (np.sqrt(subsample) - 1) # ex : for a resolution of 0.01, subsample = 10 000

        self.data_source = data_source
        self.split = split
        self.full_path = os.path.join(data_source, split)
        self.files = os.listdir(self.full_path)
        self.files = [f for f in self.files if f.endswith('.obj')]
        logging.debug(
            "using "
            + str(len(self.files))
            + " shapes from data source "
            + self.full_path
        )
        
        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.files:
                filename = os.path.join(self.full_path, f)
                if print_filename:
                    logging.info("Loading " + filename)
                data = load_data_from_file(filename, self.resolution, self.query_range)
                self.loaded_data.append(data)
                
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.load_ram:
            if type(idx) is slice:
                
                data = self.loaded_data[idx]
                start, stop, step = idx.indices(len(self.loaded_data))
                return data, [i for i in range(start, stop)]
            elif type(idx) is int:
                return self.loaded_data[idx], idx
            
        else:
            if type(idx) is slice:
                start, stop, step = idx.indices(len(self.files))
                data = [load_data_from_file(os.path.join(self.full_path, self.files[i]), self.resolution, self.query_range) for i in range(start, stop)]
                return data, [i for i in range(start, stop)]
            
            else:
                return load_data_from_file(os.path.join(self.full_path, self.files[idx]), self.resolution, self.query_range), idx
