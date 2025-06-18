import argparse
import json
import logging
import os
import torch
import numpy as np

import deep_sdf
import deep_sdf.workspace as ws

def compare_sdf(experiment_directory, checkpoint, data_source, split_filename, mesh_id):
    """
    Compares the predicted SDF from a trained DeepSDF model with the ground truth SDF values
    for a specific mesh.

    Args:
        experiment_directory (str): Path to the experiment directory containing the model and specifications.
        checkpoint (str): The checkpoint to use for the model.
        data_source (str): Path to the data source directory containing the SDF samples.
        split_filename (str): Path to the JSON file defining the data split.
        mesh_id (str): Identifier of the mesh to evaluate.
    """

    specs_filename = os.path.join(experiment_directory, "specs.json")
    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()
    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    decoder.eval()

    # Load the latent code for the mesh
    latent_vectors = ws.load_latent_vectors(experiment_directory, checkpoint)
    latent_code = latent_vectors[mesh_id].cuda()

    # Load the SDF samples for the mesh
    full_filename = os.path.join(data_source, ws.sdf_samples_subdir, mesh_id + ".npz")
    data = deep_sdf.data.read_sdf_samples_into_ram(full_filename)
    sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(data)
    xyz = sdf_data[:, 0:3].cuda()
    sdf_gt = sdf_data[:, 3].unsqueeze(1).cuda()

    # Predict SDF values using the decoder
    with torch.no_grad():
        latent_input = latent_code.expand(xyz.shape[0], -1)
        inputs = torch.cat([latent_input, xyz], dim=1).cuda()
        sdf_pred = decoder(inputs)

    # Compare predicted and ground truth SDF values
    loss_l1 = torch.nn.L1Loss()
    loss = loss_l1(sdf_pred, sdf_gt)

    logging.info(f"L1 loss between predicted and ground truth SDF: {loss.item()}")
    return loss.item(), sdf_pred.cpu().numpy(), sdf_gt.cpu().numpy(), xyz.cpu().numpy()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Compare predicted SDF with ground truth using a trained DeepSDF decoder."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory containing the model and specifications.",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint to use for the model.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        #required=True,
        help="The data source directory containing the SDF samples.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        #required=True,
        help="The split file defining the data to use.",
    )
    arg_parser.add_argument(
        "--mesh_id",
        dest="mesh_id",
        required=True,
        help="The id of the mesh to evaluate.",
    )
    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)

    specs = ws.load_experiment_specifications(args.experiment_directory)

    loss, sdf_pred, sdf_gt, xyz = compare_sdf(
        args.experiment_directory,
        args.checkpoint,
        specs["DataSource"],
        specs["TrainSplit"],
        int(args.mesh_id),
    )