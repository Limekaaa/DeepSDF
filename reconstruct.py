#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch

import deep_sdf
import deep_sdf.workspace as ws

import numpy as np
import matplotlib.pyplot as plt

import math

def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default

def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        #latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
        latent_code = torch.nn.Embedding(1, latent_size).cuda()
        torch.nn.init.normal_(latent_code.weight, mean=0.0, std=stat)
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    #latent.requires_grad = True # commented out by me

    #optimizer = torch.optim.Adam([latent], lr=lr) # commented out by me
    optimizer = torch.optim.Adam(latent_code.parameters(), lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()

        # Added by me _____________________________________________________________________
        if type(stat) == type(0.1):
            latent = latent_code(torch.tensor([0]).cuda())

        # And of added by me _____________________________________________________________________

        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).cuda()

        pred_sdf = decoder(inputs.float())
        """
        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs.float())
        """
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            print(f"loss: {loss.cpu().data.numpy()}")
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    print(f"Final loss: {loss_num}")
    #return loss_num, latent
    return loss_num, latent_code(torch.tensor([0]).cuda())  # Return the latent code from the embedding


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        #required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        #required=True,
        help="The split to reconstruct.",
    )

    arg_parser.add_argument(
        "--n_reconstructions",
        dest="n_reconstructions",
        default=20,
        help="The number of reconstructions to perform.",
    )

    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()
    
    deep_sdf.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")


    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    args.split_filename = specs["TestSplit"]
    args.data_source = specs["DataSource"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)

    random.shuffle(npz_filenames)
    npz_filenames = npz_filenames[: int(args.n_reconstructions)]
    logging.debug("reconstructing {} instances".format(len(npz_filenames)))
    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        #full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz) # Original line
        full_filename = os.path.join(args.data_source, npz) # Modified line to use ws.sdf_samples_subdir

        logging.debug("loading {}".format(npz))

        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),  # [emp_mean,emp_var],
                specs["ClampingDistance"],#0.1,
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
            )
            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()

            # Select a fixed number of evaluation points
            sdf_data_eval = deep_sdf.data.unpack_sdf_samples_from_ram(data_sdf)
            sdf_data_eval = torch.cat(sdf_data_eval, dim=0).cuda()
            eval_samples = sdf_data_eval.shape[0]

            #sdf_data_eval = deep_sdf.data.unpack_sdf_samples_from_ram(data_sdf, eval_samples).cuda()

            # Extract XYZ and ground truth SDF
            xyz_eval = sdf_data_eval[:, 0:3]
            sdf_gt_eval = sdf_data_eval[:, 3].unsqueeze(1)
            sdf_gt_eval = torch.clamp(sdf_gt_eval, -specs["ClampingDistance"], specs["ClampingDistance"])

            # Prepare input by combining xyz with optimized latent
            latent_inputs = latent.expand(eval_samples, -1)
            inputs_eval = torch.cat([latent_inputs, xyz_eval], dim=1)

            # Predict the SDF
            with torch.no_grad():
                pred_sdf_eval = decoder(inputs_eval.float())
                pred_sdf_eval = torch.clamp(pred_sdf_eval, -specs["ClampingDistance"], specs["ClampingDistance"])

            # Move everything to CPU for plotting
            xyz_np = xyz_eval[:, :2].cpu().numpy()  # Only x and y
            sdf_gt_np = sdf_gt_eval.cpu().numpy()
            pred_sdf_np = pred_sdf_eval.cpu().numpy()

            vmin = min(sdf_gt_np.min(), pred_sdf_np.min())
            vmax = max(sdf_gt_np.max(), pred_sdf_np.max())

            # Plotting
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            sc1 = axs[0].scatter(xyz_np[:, 0], xyz_np[:, 1], c=sdf_gt_np.squeeze(), cmap='viridis', vmin=vmin, vmax=vmax)
            axs[0].set_title('Ground Truth SDF')
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Y')
            plt.colorbar(sc1, ax=axs[0])

            sc2 = axs[1].scatter(xyz_np[:, 0], xyz_np[:, 1], c=pred_sdf_np.squeeze(), cmap='viridis', vmin=vmin, vmax=vmax)
            axs[1].set_title('Predicted SDF')
            axs[1].set_xlabel('X')
            axs[1].set_ylabel('Y')
            plt.colorbar(sc2, ax=axs[1])

            plt.tight_layout()

            mesh_name = os.path.splitext(os.path.basename(npz))[0]
            mesh_filename = os.path.join(reconstruction_meshes_dir, mesh_name)
            image_filename = mesh_filename + "_sdf2d.png"
            plt.savefig(image_filename)

            plt.close()

            print(f"Saved 2D SDF image to: {image_filename}")
            #plt.show()
"""
            # Define the grid resolution
            resolution = 256
            x_range = y_range = (0, 1.0)

            # Create 2D grid of points with Z = 0
            x = np.linspace(*x_range, resolution)
            y = np.linspace(*y_range, resolution)
            xx, yy = np.meshgrid(x, y)
            grid_points = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3)

            # Convert to torch tensor
            grid_tensor = torch.FloatTensor(grid_points).cuda()

            # Expand latent vector to match grid points
            latent_inputs = latent.expand(grid_tensor.shape[0], -1)

            # Concatenate latent and input points
            decoder_input = torch.cat([latent_inputs, grid_tensor], dim=1)

            # Predict SDF values
            with torch.no_grad():
                sdf_pred = decoder(decoder_input).squeeze().cpu().numpy()

            # Reshape to 2D grid
            sdf_image = sdf_pred.reshape(resolution, resolution)

            # Optional: Save SDF as image
            plt.figure(figsize=(6, 6))
            plt.imshow(sdf_image, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower', cmap='seismic')
            plt.colorbar(label='SDF value')
            plt.title('2D SDF Reconstruction (Z = 0)')
            plt.xlabel('X')
            plt.ylabel('Y')
            mesh_name = os.path.splitext(os.path.basename(npz))[0]
            mesh_filename = os.path.join(reconstruction_meshes_dir, mesh_name)
            image_filename = mesh_filename + "_sdf2d.png"
            plt.savefig(image_filename)
            plt.close()

            print(f"Saved 2D SDF image to: {image_filename}")
"""

"""
            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    deep_sdf.mesh.create_mesh(
                        decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                logging.debug("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)
"""