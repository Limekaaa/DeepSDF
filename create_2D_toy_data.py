import numpy as np
import matplotlib.pyplot as plt
import gmsh

import meshio

import math
import sys
import os

import trimesh
from tqdm import tqdm
import json 

import deep_sdf
from typing import List, Literal

def create_square_with_hole_dataset(num_samples = 100, min_hole=1, max_hole=5, min_radius=0.05, path:str="", save_format: List[Literal["msh", "stl", "obj"]] = ["msh"]):
    """
    Create a dataset of square meshes with circular holes.
    
    Parameters:
    - num_samples: Number of samples to generate.
    - min_hole: Minimum radius of the hole.
    - max_hole: Maximum radius of the hole.
    - path: Directory to save the generated meshes.
    """
    if min_radius > 0.5 * (1/max_hole) :
        raise ValueError(f"Minimum radius must be smaller than {0.5 * (1/max_hole)} for max_hole = {max_hole}.")
    
    if path[-1] != "/":
        path += "/"
    if not os.path.exists(path):
        os.makedirs(path)

    for i in tqdm(range(num_samples), desc="Generating square meshes with holes", unit="sample"):
        n_holes = np.random.randint(min_hole, max_hole + 1)
        bounds = np.linspace(0, 1, n_holes + 1)
        max_R = 0.5 * (1 / n_holes)
        hole_radii = np.random.uniform(min_radius, max_R, n_holes)

        hole_centers_bound_x = [(bounds[k] + hole_radii[k], bounds[k + 1] - hole_radii[k]) for k in range(0, n_holes, 1)]
        hole_centers_bound_y = [(hole_radii[k], 1 - hole_radii[k]) for k in range(0, n_holes, 1)]

        hole_centers = [(np.random.uniform(*hole_centers_bound_x[k]), np.random.uniform(*hole_centers_bound_y[k])) for k in range(n_holes)]

        # mesh creation ________________________________________________________________________________________________________________________________________________

        gmsh.initialize()
        gmsh.model.add("multi_circle_quad")

        # Parameters
        #lc = 0.05  # mesh size
        square_size = 1.0
        #n_circle_segments = 40  # resolution of each circular hole (more = better approximation)

        square = gmsh.model.occ.addRectangle(0, 0, 0, square_size, square_size)

        # --- 2. Circular holes
        circles = []
        for center, radius in zip(hole_centers, hole_radii):
            cx, cy = center
            circle = gmsh.model.occ.addDisk(cx, cy, 0, radius, radius)

            circles.append(circle)

        # --- 3. Plane surface with holes
        
        surface, _ = gmsh.model.occ.cut([(2, square)], [(2, c) for c in circles])

        gmsh.model.occ.synchronize()

        surf_tag = surface[0][1]
        gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        boundary = gmsh.model.getBoundary([(2, surf_tag)], oriented=False)
        curve_tags = [e[1] for e in boundary if e[0] == 1]
        gmsh.model.addPhysicalGroup(1, curve_tags, tag=2)
        gmsh.model.setPhysicalName(1, 2, "Boundary")

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.05)
        gmsh.model.mesh.generate(2)


        filename = f"square_with_holes_{n_holes}_"
        for c in hole_centers:
            filename += f"{c[0]:.3f}_"
        filename = filename[:-1]

        if "stl" in save_format:
            if not os.path.exists(f"{path}stl/"):
                os.makedirs(f"{path}stl/")
            gmsh.write(f"{path}stl/{filename}.stl")
            
        if "msh" in save_format:
            if not os.path.exists(f"{path}msh/"):
                os.makedirs(f"{path}msh/")
            gmsh.write(f"{path}msh/{filename}.msh")
            
        if "obj" in save_format:
            if "stl" not in save_format:
                print("Warning: 'obj' format requires 'stl' format to be saved as well. Saving as 'stl'.")
                if not os.path.exists(f"{path}stl/"):
                    os.makedirs(f"{path}stl/")
                gmsh.write(f"{path}stl/{filename}.stl")

            if not os.path.exists(f"{path}obj/"):
                os.makedirs(f"{path}obj/")
            tri_mesh = trimesh.load(f"{path}stl/{filename}.stl")

            trimesh.exchange.export.export_mesh(tri_mesh, f"{path}obj/{filename}.obj", file_type='obj')
        
        gmsh.finalize()



  

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Create a dataset of square meshes with circular holes.")
    parser.add_argument(
        "--experiment_directory",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include specifications file 'specs.json' and 'specs_data.json'."
    )
    args = parser.parse_args()

    
    specs_filename = os.path.join(args.experiment_directory, "specs.json")
    specs_data_filename = os.path.join(args.experiment_directory, "specs_data.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            f'The file: {specs_filename} does not exist"'
        )

    if not os.path.isfile(specs_data_filename):
        raise Exception(
            f'The file: {specs_data_filename} does not exist"'
        )

    specs = json.load(open(specs_filename))
    specs_data = json.load(open(specs_data_filename))

    if specs_data["path"][-1] != "/":
        specs_data["path"] += "/"

    full_path = specs_data["path"] + specs_data["dataset_name"] + "/"

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    if "npz" in specs_data["save_format"] and "msh" not in specs_data["save_format"]:
        print("Warning: 'npz' format requires 'msh' format to be saved as well. Saving as 'msh'.")
        specs_data["save_format"].append("msh")
        
    create_square_with_hole_dataset(
        num_samples=specs_data["num_samples"],
        min_hole=specs_data["min_hole"],
        max_hole=specs_data["max_hole"],
        min_radius=specs_data["min_radius"],
        path=full_path,
    )
    print(f"Generated {specs_data['num_samples']} samples of square meshes with circular holes in '{full_path}' directory.")

    if "npz" in specs_data["save_format"]:
        if not os.path.exists(full_path + "npz/"):
            os.makedirs(full_path + "npz/")

        samples_per_scene = specs_data["SamplesPerScene"]
        resolution = 1 / (np.sqrt(samples_per_scene) - 1)
        query_range = np.array([specs_data["query_range"]["x"], specs_data["query_range"]["y"], specs_data["query_range"]["z"]])
        print(f"Computing SDF for {len(os.listdir(full_path + 'msh/'))} meshes with samples {samples_per_scene} per scene between this range {query_range}")

        for file in os.listdir(full_path + "msh/"):
            grid_points, sdf, X, Y, Z = deep_sdf.data.compute_sdf_2d(
                os.path.join(full_path, "msh/", file),
                query_range=query_range,
                resolution=resolution
            )

            data = np.concatenate(
                [grid_points, sdf.reshape(-1, 1)], axis=1
            )

            deep_sdf.data.save_data_to_npz_file(data, filename=f"{full_path}npz/{file.replace('.msh', '')}.npz")

        npz_files = np.array(os.listdir(full_path + "npz/"))
        np.random.shuffle(npz_files)

        npz_files = [npz_files[i].replace(".npz", "") for i in range(len(npz_files))]

        train_files = npz_files[:int(specs_data["Split"]["train_proportion"] * len(npz_files))]
        test_files = npz_files[int(specs_data["Split"]["train_proportion"] * len(npz_files)):]

        if not os.path.exists(specs_data["Split"]["split_path"]):
           os.makedirs(specs_data["Split"]["split_path"])

        train_json_split = {specs_data["dataset_name"]: {"npz": train_files}}
        test_json_split = {specs_data["dataset_name"]: {"npz": test_files}}

        json.dump(train_json_split, open(os.path.join(specs_data["Split"]["split_path"], f"{specs_data['dataset_name']}_train.json"), "w"))
        json.dump(test_json_split, open(os.path.join(specs_data["Split"]["split_path"], f"{specs_data['dataset_name']}_test.json"), "w"))