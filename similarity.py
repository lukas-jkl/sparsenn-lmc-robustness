import copy
from typing import List, Dict, OrderedDict

import torch
import pandas as pd
import numpy as np
import pickle as pkl
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

from permute import interpolate_state_dicts
from model.builder import build_model
import utils.util as util


def center_columns(mat: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Compute the column average and subtract it
    n = mat.shape[0]
    H = torch.eye(n) - torch.ones([n, n]) / n  # H is the centering matrix
    return torch.matmul(H.to(device), mat)


def linear_cka(x: torch.Tensor, y: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = center_columns(x, device)
    y = center_columns(y, device)
    return torch.norm(x.T @ y, 'fro') ** 2 / \
        (torch.norm(x.T @ x, 'fro') * torch.norm(y.T @ y, 'fro'))


def cosine_sim_squared(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (torch.dot(x, y) / (torch.norm(x) * torch.norm(y))) ** 2


def compute_model_similarity_measures(model_1: nn.Sequential, model_2: nn.Sequential, data_loader: DataLoader,
                                      device: torch.device) -> pd.DataFrame:
    def get_activation(activations, i):
        def hook(model, input, output):
            activations[i] = output.detach()
        return hook

    hooks = []
    model_i_activations, model_1_activations = {}, {}
    cka_vals = {}
    measures = pd.DataFrame()
    for layer_idx, (model_i_layer, model_1_layer) in enumerate(zip(list(model_2.children()), list(model_1.children()))):
        try:
            # Compute cosine sim
            model_i_weights = torch.cat([torch.flatten(model_i_layer.weight), model_i_layer.bias], dim=0)
            model_1_weights = torch.cat([torch.flatten(model_1_layer.weight), model_1_layer.bias], dim=0)
            cosine_sim = cosine_sim_squared(model_i_weights, model_1_weights).item()
            measures = measures.append({"measure": "cosine_sim", "value": cosine_sim, "layer": layer_idx},
                                       ignore_index=True)

            # Add hooks to capture activations
            hooks.append(model_i_layer.register_forward_hook(get_activation(model_i_activations, layer_idx)))
            hooks.append(model_1_layer.register_forward_hook(get_activation(model_1_activations, layer_idx)))

            cka_vals[layer_idx] = []
        except AttributeError:
            pass  # no layer weights

    sum_pred_diff = 0
    for data, target in data_loader:
        data, target = data.to(device).view(data.size(0), -1), target.to(device)

        # Compute outputs
        model_1_output = model_1(data)
        model_2_output = model_2(data)

        # Compute pred disagreement
        model_2_pred = model_2_output.max(1)[1]
        model_1_pred = model_1_output.max(1)[1]
        sum_pred_diff += model_2_pred.ne(model_1_pred).sum().item()

        # Compute CKA
        for layer_idx in cka_vals.keys():
            cka_vals[layer_idx].append(
                linear_cka(model_i_activations[layer_idx], model_1_activations[layer_idx], device).item())

    # Store results
    for layer_idx in cka_vals.keys():
        cka = torch.mean(torch.tensor(cka_vals[layer_idx])).item()
        measures = measures.append({"measure": "cka", "value": cka, "layer": layer_idx}, ignore_index=True)
    measures = measures.append(
        {"measure": "pred_disag", "value": (sum_pred_diff / len(data_loader.dataset)) * 100, "layer": None},
        ignore_index=True)

    [hook.remove() for hook in hooks]
    return measures


def measure_valley_tightness(models: List[torch.nn.Sequential], device: torch.device, data_loader: DataLoader,
                             std_dev_values: List[float], num_samples: int = 10, noise_type="model_weights_abs") \
                                  -> Dict:
    criterion = nn.CrossEntropyLoss()
    accuracy_measures = np.zeros((len(std_dev_values), len(models)))
    loss_measures = np.zeros((len(std_dev_values), len(models)))

    cka_measures = {}
    cka_model_1_measues = {}
    for idx, layer in enumerate(list(models[0].children())):
        try:
            _ = layer.weight
            cka_measures[idx] = np.zeros((len(std_dev_values), len(models)))
            cka_model_1_measues[idx] = np.zeros((len(std_dev_values), len(models)))
        except AttributeError:
            pass

    model_1 = models[0]
    for model_idx, model_i in enumerate(tqdm(models, leave=False)):
        for std_idx, std_dev in enumerate(tqdm(std_dev_values, leave=False)):
            for _ in range(num_samples):
                noisy_model = copy.deepcopy(model_i)
                noisy_data_loader = data_loader
                if std_dev != 0:
                    if "model" in noise_type:
                        with torch.no_grad():
                            util.apply_noise_to_model(noisy_model, std_dev, noise_type)
                    elif "data" in noise_type:
                        noisy_data_loader = util.get_noisy_data_loader(data_loader, device, 0, std_dev)
                    else:
                        raise Exception("Invalid noise_type:", noise_type)

                m = util.evaluate_model(noisy_model, device, noisy_data_loader, criterion, True)
                loss_measures[std_idx][model_idx] += m["Loss"]
                accuracy_measures[std_idx][model_idx] += m["Accuracy"]

                sim_measures = compute_model_similarity_measures(model_i, noisy_model, data_loader, device)
                sim_measures = sim_measures[sim_measures.measure == "cka"]
                sim_measures_model_1 = compute_model_similarity_measures(model_1, noisy_model, data_loader, device)
                sim_measures_model_1 = sim_measures_model_1[sim_measures_model_1.measure == "cka"]
                for layer_idx in sim_measures.layer.dropna().unique():
                    cka_measures[layer_idx][std_idx][model_idx] += \
                        sim_measures.loc[sim_measures.layer == layer_idx].value.values[0]
                    cka_model_1_measues[layer_idx][std_idx][model_idx] += \
                        sim_measures_model_1.loc[sim_measures_model_1.layer == layer_idx].value.values[0]

    for key in cka_measures:
        cka_measures[key] /= num_samples
    for key in cka_model_1_measues:
        cka_model_1_measues[key] /= num_samples
    accuracy_measures /= num_samples
    loss_measures /= num_samples

    return {
        "std_dev_values": std_dev_values,
        "accuracy_measures": accuracy_measures,
        "loss_measures": loss_measures,
        "cka_measures": cka_measures,
        "cka_model_1_measues": cka_model_1_measues,
    }


def calculate_interpolated_model_measures(model_1: nn.Sequential, permuted_model_state_dicts: List[OrderedDict],
                                          data_loader: DataLoader, device: torch.device,
                                          alpha_step=0.1) -> pd.DataFrame:
    alpha_start = 0
    alpha_end = 1 + alpha_step
    model_1.to(device)
    model = copy.deepcopy(model_1)
    alphas = np.round(np.arange(alpha_start, alpha_end, alpha_step), 2)
    criterion = nn.CrossEntropyLoss().to(device)

    measures = pd.DataFrame()
    for i, permuted_model in enumerate(tqdm(permuted_model_state_dicts, leave=False)):
        for alpha in alphas:
            interpolated_state_dict = interpolate_state_dicts(model_1.state_dict(), permuted_model, alpha)
            model.load_state_dict(interpolated_state_dict)
            model.to(device)
            sim_measures = compute_model_similarity_measures(model, model_1, data_loader, device)
            eval_train = util.evaluate_model(model, device, data_loader, criterion, reshape_input=True)
            for key, value in eval_train.items():
                sim_measures = sim_measures.append({"measure": key, "value": value}, ignore_index=True)
            sim_measures["pair"] = i
            sim_measures["alpha"] = alpha
            measures = measures.append(sim_measures)
    return measures


def run_measure_valley_tightness():
    parser = argparse.ArgumentParser(
        description='Run similarity measure computation for ensemble, only works for model arch MLP_no_flatten.')
    parser.add_argument('--cuda_device_num', default=0, type=int,
                        help='which cuda device to use')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='Path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='Name of the dataset (options: MNIST, default: MNIST)')
    parser.add_argument('--nunits', type=int,
                        help='Number of hidden units', required=True),
    parser.add_argument('--nlayers', type=int,
                        help='Number of hidden layers', default=1),
    parser.add_argument('--nsamples', default=32, type=int,
                        help='The number of random samples taken for the perturbed models, (default: 32)')
    parser.add_argument('--inputfile', type=str,
                        help='The file from which the results are loaded. (results.pkl file)', required=True)
    parser.add_argument('--outputpath', default="experiment_results/computed_measures/", type=str,
                        help='Path where the results are stored.')
    parser.add_argument('--google-cloud', default=False, action='store_true',
                        help='Enables upload/download from google cloud')
    parser.add_argument("--noise_type", default="model_weights_abs", type=str,
                        help="The noise to apply (default: model_weight_abs)")

    args = parser.parse_args()
    google_cloud_bucket = "lukas_deep"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(args.cuda_device_num)

    # Load data
    if args.dataset == 'MNIST':
        nchannels = 1
        nclasses = 10
    elif args.dataset == 'CIFAR10':
        nchannels = 3
        nclasses = 10
    elif args.dataset == 'CIFAR100':
        nchannels = 3
        nclasses = 100
    else:
        raise Exception("Unknown Dataset")

    test_dataset = util.load_data("test", args.dataset, args.datadir, nchannels)
    test_loader = util.get_dataset_loader(test_dataset, batch_size=1000)

    # Load models
    model = build_model("MLP_no_flatten", nchannels, nclasses, args.nunits, args.nlayers)
    model.to(device)
    if args.google_cloud:
        util.download_blob_if_not_exists(google_cloud_bucket, args.inputfile, args.inputfile)
    all_results = util.load_gpu_pkl(args.inputfile, device)

    models = []
    results = all_results[all_results.job == all_results.job.unique()[0]]  # only check for one job
    for state_dict in results.current_model_state_dict:
        model_i = copy.deepcopy(model)
        model_i.load_state_dict(state_dict)
        models.append(model_i)

    # Compute measures
    data_loader = test_loader
    std_dev_values = list(np.arange(0, 2.4, 0.4))
    outputpath = os.path.join(args.outputpath, "measure_valley_tightness_ensemble_" + args.noise_type)
    outputfile = os.path.join(outputpath, os.path.basename(os.path.dirname(args.inputfile)) + ".pkl")
    result = measure_valley_tightness(models, device, data_loader, std_dev_values, args.nsamples, args.noise_type)

    # Save to file
    outdir = os.path.dirname(outputfile)
    os.makedirs(outdir, exist_ok=True)
    with open(outputfile, "wb") as f:
        pkl.dump(result, f, protocol=4)
    if args.google_cloud:
        util.upload_blob(google_cloud_bucket, outputfile, outputfile)
    print("Written results to file ", outputfile)


if __name__ == '__main__':
    run_measure_valley_tightness()
