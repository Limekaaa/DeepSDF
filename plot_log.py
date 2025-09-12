#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import deep_sdf
import deep_sdf.workspace as ws
import json

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def load_logs(experiment_directory, type):

    logs = torch.load(os.path.join(experiment_directory, ws.logs_filename), weights_only = False)
    specs = json.load(open(os.path.join(experiment_directory, "specs.json")))
    test_freq = specs.get("TestFrequency", 100)
    logging.info("latest epoch is {}".format(logs["epoch"]))

    num_iters = len(logs["loss"])
    iters_per_epoch = num_iters / logs["epoch"]

    logging.info("{} iters per epoch".format(iters_per_epoch))

    smoothed_loss_41 = running_mean(logs["loss"], 41)
    smoothed_loss_1601 = running_mean(logs["loss"], 1601)

    fig, ax = plt.subplots()

    if type == "loss":

        ax.plot(
            np.arange(num_iters) / iters_per_epoch,
            logs["loss"],
            "#82c6eb",
            np.arange(20, num_iters - 20) / iters_per_epoch,
            smoothed_loss_41,
            "#2a9edd",
            np.arange(800, num_iters - 800) / iters_per_epoch,
            smoothed_loss_1601,
            "#16628b",
        )

        # Adding test loss data to the plot
        #test_epochs = np.arange(0, logs["epoch"], test_freq) #/ iters_per_epoch  # Epochs corresponding to test loss logs
        #test_epochs = test_epochs[: len(logs['test_loss'])]  # Ensure matching lengths

        test_loss = np.array(logs['test_loss'])
        test_epochs = np.array([i*test_freq for i in range(1,len(test_loss)+1)])
        test_loss = np.interp(np.arange(logs["epoch"]), test_epochs, test_loss)
        
        ax.plot(
            np.arange(logs["epoch"]),
            test_loss,
            color="#ff7f0e",  # Orange color for test loss
            label="Test Loss",
            #marker='o',  # Optional: Adding markers for clarity
        )
        #print(np.arange(test_freq, logs["epoch"], test_freq))
        ax.plot(
            np.arange(test_freq, logs["epoch"]+test_freq, test_freq),
            logs["test_loss"],
            color="#ff7f0e",  # Orange color for test loss
            label="Test Loss",
            marker='o',  # Optional: Adding markers for clarity
        )

        # Setting labels and title
        ax.set(xlabel="Epoch", ylabel="Loss", title="Training and Test Loss Over Time")

    elif type == "learning_rate":
        combined_lrs = np.array(logs["learning_rate"])

        ax.plot(
            np.arange(combined_lrs.shape[0]),
            combined_lrs[:, 0],
            np.arange(combined_lrs.shape[0]),
            combined_lrs[:, 1],
        )
        ax.set(xlabel="Epoch", ylabel="Learning Rate", title="Learning Rates")

    elif type == "time":
        ax.plot(logs["timing"], "#833eb7")
        ax.set(xlabel="Epoch", ylabel="Time per Epoch (s)", title="Timing")

    elif type == "lat_mag":
        ax.plot(logs["latent_magnitude"])
        ax.set(xlabel="Epoch", ylabel="Magnitude", title="Latent Vector Magnitude")

    elif type == "param_mag":
        for _name, mags in logs["param_magnitude"].items():
            ax.plot(mags)
        ax.set(xlabel="Epoch", ylabel="Magnitude", title="Parameter Magnitude")
        ax.legend(logs["param_magnitude"].keys())
    elif type == 'all':
        types = ['loss', 'learning_rate', 'time', 'lat_mag', 'param_mag']
        for t in types:
            load_logs(experiment_directory, t)

    else:
        raise Exception('unrecognized plot type "{}"'.format(type))

    if type != 'all':
        ax.grid()
        if not os.path.exists(f"{experiment_directory}/LogsFigs"):
            os.makedirs(f"{experiment_directory}/LogsFigs")
        
        plt.savefig(f"{experiment_directory}/LogsFigs/{type}.png")
        if type == "loss":
            plt.yscale("log")
            plt.savefig(f"{experiment_directory}/LogsFigs/{type}_log.png")
        plt.close()
        #plt.show()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Plot DeepSDF training logs")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment "
        + "specifications in 'specs.json', and logging will be done in this directory "
        + "as well",
    )
    arg_parser.add_argument("--type", "-t", dest="type", default="loss")

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    load_logs(args.experiment_directory, args.type)
