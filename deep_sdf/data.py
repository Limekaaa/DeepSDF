#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import pytorch_volumetric as pv

import deep_sdf.workspace as ws


import os
import numpy as np
import meshio
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# Added by me _______________________________________________________________________________________________________

def compute_sdf_2d(mesh_path, query_range=np.array([[0, 1], [0, 1], [0, 0]]), resolution=0.01):

    mesh = meshio.read(mesh_path)
    points = mesh.points[:, :2]  # Use only the first two dimensions for 2D SDF

    boundary_edges = []
    for cell_block in mesh.cells:
        if cell_block.type in ["line", "line2"]:
            for edge in cell_block.data:
                boundary_edges.append(points[edge[0]])
                boundary_edges.append(points[edge[1]])
    if not boundary_edges:
        raise ValueError(f"No boundary edges found in {mesh_path}")
    boundary_points = np.unique(np.vstack(boundary_edges), axis=0)
    tree = cKDTree(boundary_points)

    # Sample grid
    # x = np.linspace(-0.1, 1.1, grid_size)
    # y = np.linspace(-0.1, 1.1, grid_size)
    
    # The bounding box
    x = np.arange(query_range[0][0], query_range[0][1]+resolution, resolution)
    y = np.arange(query_range[1][0], query_range[1][1]+resolution, resolution)
    #z = np.arange(query_range[2][0], query_range[2][1], resolution)

    #X, Y, Z = np.meshgrid(x, y, z)
    X, Y = np.meshgrid(x, y)
    grid_points = np.c_[X.ravel(), Y.ravel()]
    distances, _ = tree.query(grid_points)

    # Create polygon union of all triangles to test for inside
    polygons = []
    for cell_block in mesh.cells:
        if cell_block.type.startswith("triangle"):
            for tri in cell_block.data:
                polygons.append(Polygon(points[tri]))
    domain = unary_union(polygons)

    signs = np.array([1 if domain.contains(Point(p)) else -1 for p in grid_points])
    sdf = distances * signs

    Z = np.zeros_like(X)  # Z is zero for 2D case
    grid_points = np.c_[grid_points, Z.ravel()]  # Add Z coordinate
    return grid_points, sdf, X, Y, Z

def save_data_to_npz_file(data:np.ndarray, filename:str):
    """
    Save data to npz files

    Args:
        data (np.ndarray): data of shape (N, 4) where N is the number of points and 4 represents (x, y, z, sdf_value).
        filename (str): Path to the output npz file.
    """

    pos = data[data[:, -1] >= 0]
    neg = data[data[:, -1] < 0]

    np.savez(filename, pos=pos, neg=neg)
    
# __________________________________________________________________________________________________________________________________________



def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
    
    if subsample == pos_tensor.shape[0] + neg_tensor.shape[0]:
        # If subsample is equal to the total number of samples, return all samples
        return torch.cat([pos_tensor, neg_tensor], 0).float()
    # split the sample into half
    half = int(subsample / 2)

    if pos_tensor.shape[0] < half:
        sample_pos = pos_tensor
        to_complete = half - pos_tensor.shape[0]
        random_neg = (torch.rand(to_complete+half) * neg_tensor.shape[0]).long()

        sample_neg = torch.index_select(neg_tensor, 0, random_neg)#[:to_complete+half]

    elif neg_tensor.shape[0] < half:
        sample_neg = neg_tensor
        to_complete = half - neg_tensor.shape[0]
        random_pos = (torch.rand(to_complete+half) * pos_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)

    else:
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0).float()

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    if subsample == pos_tensor.shape[0] + neg_tensor.shape[0]:
        # If subsample is equal to the total number of samples, return all samples
        return torch.cat([pos_tensor, neg_tensor], 0).float()
    # split the sample into half
    half = int(subsample / 2)

    if pos_tensor.shape[0] < half:
        sample_pos = pos_tensor
        to_complete = half - pos_tensor.shape[0]
        random_neg = (torch.rand(to_complete+half) * neg_tensor.shape[0]).long()

        sample_neg = torch.index_select(neg_tensor, 0, random_neg)#[:to_complete+half]

    elif neg_tensor.shape[0] < half:
        sample_neg = neg_tensor
        to_complete = half - neg_tensor.shape[0]
        random_pos = (torch.rand(to_complete+half) * pos_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)

    else:
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0).float()

    return samples


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
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            #self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx] # original line
            self.data_source, self.npyfiles[idx] # modified line to match the new structure
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx
