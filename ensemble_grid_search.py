import argparse
import copy
import os
import random
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import pickle as pkl
import hashlib

import utils.util as util
import permute
from model.builder import build_model


def ensemble_grid_search(start_model: torch.nn.Sequential, model_pool: List[torch.nn.Sequential], device: torch.device,
                         train_dataset: Dataset, test_dataset: Dataset, ensemble_size: int = 3,
                         grid_search_parameters: Dict = None, annealing_data_size: int = 10000,
                         result_folder: str = "checkpoints",
                         google_cloud: bool = False, google_cloud_bucket: str = "lukas_deep"):
    if grid_search_parameters is not None:
        tmax_values = grid_search_parameters["tmax_values"]
        tmin_values = grid_search_parameters["tmin_values"]
        step_values = grid_search_parameters["step_values"]
        iterations = grid_search_parameters["iterations"]
    else:
        # If no grid search parameters we use the automatic scheduler
        tmax_values, tmin_values, step_values, iterations = [None], [None], [None], 1

    # Only use part of training data for annealing
    partial_train_loader = util.get_dataset_loader(train_dataset, batch_size=annealing_data_size,
                                                   samples=annealing_data_size)

    # Use full dataset for verifying train and test accuracy
    full_train_loader = util.get_dataset_loader(train_dataset, batch_size=1000)
    test_loader = util.get_dataset_loader(test_dataset, batch_size=1000)

    measures = pd.DataFrame()
    results = pd.DataFrame()
    current_model = copy.deepcopy(start_model)
    interpolated_model = copy.deepcopy(start_model)

    current_best_results = {
        "ensemble_size": 1,
        "barrier": 0,
        "annealing_parameters": {},
        "train_accuracy": util.evaluate_model(start_model, device, full_train_loader, reshape_input=True)["Accuracy"],
        "test_accuracy": util.evaluate_model(start_model, device, test_loader, reshape_input=True)["Accuracy"],
    }
    print("\nCurrent results: ", current_best_results)
    current_best_results["current_model_state_dict"] = start_model.state_dict()
    results = results.append(current_best_results, ignore_index=True)
    used_model_indices = []
    os.makedirs(result_folder, exist_ok=True)

    # Perform grid search:
    with tqdm(total=len(step_values) * len(tmax_values) * len(tmin_values) * (ensemble_size - 1) * iterations) as pbar:
        for i in range(2, ensemble_size + 1):
            current_highest_train_acc = 0
            current_best_model, current_used_model_idx = None, None
            for steps in step_values:
                for tmax in tmax_values:
                    for tmin in tmin_values:
                        for _ in range(iterations):
                            # Pick random model from model-pool which has not yet been used
                            model_idx = random.choice(
                                list(filter(lambda x: x not in used_model_indices, range(len(model_pool)))))

                            if grid_search_parameters is not None:
                                annealing_params = {"tmax": tmax, "tmin": tmin, "steps": steps, "updates": 0}
                            else:
                                annealing_params = None

                            # Find permuted model
                            perm_model, e = permute.simaneal_find_permuted_model(model_pool[model_idx],
                                                                                 current_model, device,
                                                                                 partial_train_loader,
                                                                                 annealing_params)

                            # Evaluate model between permuted and current-model
                            interpolated_model.load_state_dict(
                                permute.interpolate_state_dicts(current_model.state_dict(), perm_model.state_dict()))
                            train_acc = util.evaluate_model(interpolated_model, device, full_train_loader,
                                                            reshape_input=True)["Accuracy"]
                            test_acc = util.evaluate_model(interpolated_model, device, test_loader,
                                                           reshape_input=True)["Accuracy"]

                            # Append measures
                            current_results = {
                                "ensemble_size": i,
                                "used_model_idx": model_idx,
                                "barrier": e,
                                "annealing_parameters": annealing_params,
                                "train_accuracy": train_acc,
                                "test_accuracy": test_acc,
                                "annealing_data_size": annealing_data_size
                            }
                            print("\nCurrent results: ", current_results)
                            measures = measures.append(current_results, ignore_index=True)

                            # Update best model found in grid search
                            if train_acc > current_highest_train_acc:
                                current_highest_train_acc = train_acc
                                current_used_model_idx = model_idx
                                current_best_results = current_results
                                best_p_model_state_dict = perm_model.state_dict()
                                current_best_model = copy.deepcopy(interpolated_model)
                            pbar.update(1)

            # Update current model to best model found for current ensemble size
            current_model = current_best_model
            used_model_indices.append(current_used_model_idx)  # remember which models from pool already used

            # Save results for current ensemble size
            current_best_results["p_model_state_dict"] = best_p_model_state_dict
            current_best_results["current_model_state_dict"] = current_best_model.state_dict()
            results = results.append(current_best_results, ignore_index=True)

            measure_file = result_folder + "/{}_measures{}.pkl".format(i, "_final" if i == ensemble_size else "")
            result_file = result_folder + "/{}_results{}.pkl".format(i, "_final" if i == ensemble_size else "")

            if google_cloud:
                result_dump = pkl.dumps(results, protocol=4)
                util.upload_pkl(google_cloud_bucket, result_dump, result_file)
                measure_dump = pkl.dumps(measures, protocol=4)
                util.upload_pkl(google_cloud_bucket, measure_dump, measure_file)
            else:
                results.to_pickle(result_file, protocol=4)
                measures.to_pickle(measure_file, protocol=4)

    return results, measures


def main():
    # settings
    parser = argparse.ArgumentParser(description="""
                                         Ensemble finder, builds ensemble of models with arch MLP_no_flatten.
                                         Performs grid search over simmulated annealing parameters and takes
                                          the best found models.""")
    parser.add_argument('--cuda_device_num', default=0, type=int,
                        help='Which cuda device to use')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='Path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--modeldir', required=True, type=str,
                        help='Path to the directory that contains the models')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='Name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: MNIST)')
    parser.add_argument('--nunits', type=int, required=True, help='number of hidden units')
    parser.add_argument('--nlayers', type=int, default=1, help='number of hidden layers')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--tmax_values', default=[25], type=float, nargs='+',
                        help='List of tmax values to try e.g. 30 24 12, default: 25')
    parser.add_argument('--tmin_values', default=[0.001], type=float, nargs='+',
                        help='List of tmin values to try e.g. 0.005 0.001, default: 0.001')
    parser.add_argument('--step_values', default=[50000], type=int, nargs='+',
                        help='List of step values to try for annealing e.g. 50000 70000, default: 50000')
    parser.add_argument('--iterations', default=1, type=int,
                        help='The number of times annealing with the current parameters is performed, default: 1')
    parser.add_argument('--train_data_size', default=60000, type=int,
                        help='Portion of the training data used for annealing in number of samples, default: 60000')
    parser.add_argument('--ensemble_size', default=3, type=int,
                        help='Size of the ensemble that shall be created, default: 3')
    parser.add_argument('--model_pool_size', default=10, type=int,
                        help='Number of models that can be loaded from modeldir, default 10')
    parser.add_argument('--auto_find_annealing_params', default=False, action='store_true',
                        help='Ignore all annealing parameters and instead search for them automatically')
    parser.add_argument('--google-cloud', default=False, action='store_true',
                        help='Runs with upload and downloads to google cloud, default: False')
    parser.add_argument('--resultdir', type=str, help='The directory in which to save the results', required=True)
    args = parser.parse_args()

    google_cloud_bucket = "lukas_deep"

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(args.cuda_device_num)

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

    if args.google_cloud:
        util.download_folder(google_cloud_bucket, args.modeldir)
    model_files = util.get_model_files_from_dir(args.modeldir, args.model_pool_size)

    # Load models
    model = build_model("MLP_no_flatten", nchannels, nclasses, args.nunits, args.nlayers)
    model = model.to(device)
    model_1 = None
    model_2_s = []
    model_files.sort()
    for i, model_file in enumerate(model_files):
        model_i = copy.deepcopy(model)
        model_i.load_state_dict(torch.load(model_file, map_location=device))
        if i + 1 == 1:
            model_1 = model_i
        else:
            model_2_s.append(model_i)

    if model_1 is None or len(model_2_s) < args.ensemble_size - 1:
        raise Exception("Not enough models for ensemble")
    print("Found {} models in directory to ensemble with model_1".format(len(model_2_s)))

    # Load data
    train_dataset = util.load_data("train", args.dataset, args.datadir, nchannels)
    test_dataset = util.load_data("test", args.dataset, args.datadir, nchannels)

    if not args.auto_find_annealing_params:
        grid_search_parameters = {"tmax_values": args.tmax_values,
                                  "tmin_values": args.tmin_values,
                                  "step_values": args.step_values,
                                  "iterations": args.iterations}
    else:
        grid_search_parameters = None

    # Generate subfolder based on args + random so we don't overwrite previous results
    sha = hashlib.md5(str(args).encode())
    result_dir = args.resultdir + "/" + sha.hexdigest() + str(random.randint(0, 10000))

    # Run grid search
    ensemble_grid_search(model_1, model_2_s, device, train_dataset, test_dataset, args.ensemble_size,
                         grid_search_parameters,
                         annealing_data_size=args.train_data_size,
                         result_folder=result_dir,
                         google_cloud=args.google_cloud,
                         google_cloud_bucket=google_cloud_bucket)


if __name__ == '__main__':
    main()
