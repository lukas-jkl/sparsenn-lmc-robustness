import argparse
import torch
import torch.nn as nn
import copy
import random
from simanneal import Annealer
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from tqdm import tqdm
import pandas as pd
import os
from model.builder import build_model

import utils.util as util


class ModelAnnealer(Annealer):
    def __init__(self, state, original_model: nn.Sequential, device: torch.device, data_loader: DataLoader):
        super(ModelAnnealer, self).__init__(state)
        self.original_weights = []
        for child in original_model.children():
            if child._get_name() == "Linear":
                self.original_weights.append(torch.clone(child.weight.data.detach()))
                self.original_weights.append(torch.clone(child.bias.data.detach()))

        self.nunits = self.original_weights[0].shape[0]
        self.data_loader = data_loader
        self.inputs, self.targets = util.unpack_dataloader(data_loader, device, reshape_input=True)
        self.device = device

    def move(self):
        initial_energy = self.energy()
        for i in range(len(self.state)):
            x = self.state[i]
            a = random.randint(0, len(x) - 1)
            b = random.randint(0, len(x) - 1)
            self.state[i][a], self.state[i][b] = self.state[i][b], self.state[i][a]
        return self.energy() - initial_energy

    def permute(self, permutation):
        return _permute(permutation, self.original_weights, self.nunits, self.device)


class LmcAnnealer(ModelAnnealer):
    def __init__(self, state, original_model: nn.Sequential, lmc_target_model: nn.Sequential, device: torch.device,
                 data_loader: DataLoader):
        super(LmcAnnealer, self).__init__(state, original_model, device, data_loader)
        lmc_target_model.to(device)
        self.sd1 = copy.deepcopy(lmc_target_model).state_dict()
        model1_eval = util.evaluate_model(original_model, device, data_loader, reshape_input=True)["Accuracy"]
        model2_eval = util.evaluate_model(lmc_target_model, device, data_loader, reshape_input=True)["Accuracy"]
        self.comparison_accuracy = (model1_eval + model2_eval) / 2
        self.model = copy.deepcopy(original_model)
        self.model.to(device).eval()

    def energy(self):
        permuted_model = self.permute(self.state)
        permuted_model.to(self.device)
        self.model.load_state_dict(interpolate_state_dicts(self.sd1, permuted_model.state_dict(), 0.5))
        correct = 0
        with torch.no_grad():
            for data, target in zip(self.inputs, self.targets):
                correct += self.model(data).max(1)[1].eq(target).sum().item()

        lmc_accuracy = (correct / len(self.data_loader.dataset)) * 100
        e = self.comparison_accuracy - lmc_accuracy
        return e


def _permute(perm_ind, w_2, nunits, device):
    [a.to(device) for a in w_2]

    # 1 layer
    if len(w_2) == 4:
        assert len(perm_ind) == 1
        # permutation
        idx = perm_ind
        # permute weights of model2, based on idx
        w1 = w_2[0]
        b1 = w_2[1]
        w2 = w_2[2]
        b2 = w_2[3]

        w1_p = w1[idx, :]
        b1_p = b1[idx]
        w2_p = w2[:, idx]
        b2_p = b2

        nchannels, nclasses = 1, 10
        model = build_model("MLP_no_flatten", nchannels, nclasses, nunits, 1)
        model.to(device).eval()

        model.state_dict()["0.weight"][:] = w1_p
        model.state_dict()["0.bias"][:] = b1_p
        model.state_dict()["2.weight"][:] = w2_p.squeeze()
        model.state_dict()["2.bias"][:] = b2_p
    elif len(w_2) == 6:
        assert len(perm_ind) == 2
        w1 = w_2[0]
        b1 = w_2[1]
        w2 = w_2[2]
        b2 = w_2[3]
        w3 = w_2[4]
        b3 = w_2[5]

        idx1 = perm_ind[0]
        w1_p = w1[idx1, :]
        b1_p = b1[idx1]

        idx2 = perm_ind[1]
        w2_p = w2[:, idx1]    # to take care of prv permutation
        w2_p = w2_p[idx2, :]  # to apply new permutation
        b2_p = b2[idx2]

        idx2 = perm_ind[1]
        w3_p = w3[:, idx2]
        b3_p = b3

        nchannels, nclasses = 1, 10
        model = build_model("MLP_no_flatten", nchannels, nclasses, nunits, 2)
        model = model.to(device)

        model.state_dict()["0.weight"][:] = w1_p
        model.state_dict()["0.bias"][:] = b1_p
        model.state_dict()["2.weight"][:] = w2_p.squeeze()
        model.state_dict()["2.bias"][:] = b2_p
        model.state_dict()["4.weight"][:] = w3_p.squeeze()
        model.state_dict()["4.bias"][:] = b3_p
    else:
        raise NotImplementedError("Architecture not supported.")
    return model


def interpolate_state_dicts(state_dict_1, state_dict_2, weight=0.5):
    return {key: (1 - weight) * state_dict_1[key] + weight * state_dict_2[key]
            for key in state_dict_1.keys()}


def simaneal_find_permuted_model(original_model: torch.nn.Sequential, lmc_target_model: torch.nn.Sequential,
                                 device: torch.device, data_loader: DataLoader,
                                 annealing_params: Dict) -> Tuple[nn.Sequential, float]:
    num_units = list(original_model.children())[0].weight.shape[0]
    num_hidden_layers = round(len(original_model.state_dict().keys()) / 2) - 1
    init_state = [list(range(num_units)) for _ in range(num_hidden_layers)]
    [random.shuffle(s) for s in init_state]
    annealer = LmcAnnealer(init_state, original_model, lmc_target_model, device, data_loader)

    if not annealing_params:
        annealing_params = annealer.auto(minutes=15, steps=2000)
        print("annealing_params:", annealing_params)

    annealer.set_schedule(annealing_params)
    winning_permutation, e = annealer.anneal()
    winning_perm_model = annealer.permute(winning_permutation)
    return winning_perm_model, e


def main():
    parser = argparse.ArgumentParser(
        description='Take first model (model 1) and permute the remaining models to be LMC to model 1.'
        + ' For models with arch MLP_no_flatten.'
        )
    parser.add_argument('--cuda_device_num', default=0, type=int,
                        help='which cuda device to use')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='Path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='Name of the dataset (options: MNIST | CIFAR10 | CIFAR100, default: MNIST)')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--modeldir', type=str,
                        help='The directory from which the models are loaded', required=True)
    parser.add_argument('--model_pool_size', default=10, type=int,
                        help='Number of models that can be loaded from modeldir, default 10')
    parser.add_argument('--nunits', type=int,
                        help='Number of hidden units', required=True),
    parser.add_argument('--nlayers', type=int,
                        help='Number of hidden layers', default=1),
    parser.add_argument('--tmax', default=25, type=float,
                        help='Tmax value used for annealing, default: 25')
    parser.add_argument('--tmin', default=0.001, type=float,
                        help='Tmin value used for annealing, default: 0.001')
    parser.add_argument('--steps', default=50000, type=int,
                        help='Steps used for annealing, default: 50000')
    parser.add_argument('--iterations', default=1, type=int,
                        help='The number of times annealing with the current parameters is performed, default: 1')
    parser.add_argument('--train_data_size', default=60000, type=int,
                        help='Portion of the training data used for annealing in number of samples, default: 60000')
    parser.add_argument('--auto_find_annealing_params', default=False, action='store_true',
                        help='Ignore all annealing parameters and instead search for them automatically')
    parser.add_argument('--google-cloud', default=False, action='store_true',
                        help='Enables upload/download from google cloud')
    parser.add_argument('--resultdir', type=str, help='The directory in which to save the result', required=True)

    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
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

    train_dataset = util.load_data("train", args.dataset, args.datadir, nchannels)
    train_loader = util.get_dataset_loader(train_dataset, batch_size=1000, samples=None)
    partial_train_loader = util.get_dataset_loader(train_dataset, batch_size=1000, samples=args.train_data_size)

    test_dataset = util.load_data("test", args.dataset, args.datadir, nchannels)
    test_loader = util.get_dataset_loader(test_dataset, batch_size=1000)

    # Load models
    model_files = util.get_model_files_from_dir(args.modeldir, args.model_pool_size)
    model_files.sort()
    model_1 = None
    model_2_s = []
    for i, model_file in enumerate(model_files):
        try:
            sd = torch.load(model_file, map_location=device)
            model_i = build_model("MLP_no_flatten", nchannels, nclasses, args.nunits, args.nlayers)
            model_i = model_i.to(device)
            model_i.load_state_dict(sd)
            model_i.to(device).eval()
            if i + 1 == 1:
                model_1 = model_i
            else:
                model_2_s.append(model_i)
        except FileNotFoundError:
            break  # stop searching for next model

    if model_1 is None or len(model_2_s) < 1:
        print("Did not find enough models for ensemble")
        raise Exception()
    else:
        print("Found {} models in directory to permute and model_1".format(len(model_2_s)))

    # Set annealing parameters
    if args.auto_find_annealing_params:
        annealing_params = None
    else:
        annealing_params = {
            "tmax": args.tmax,
            "tmin": args.tmin,
            "steps": args.steps,
            "updates": 100
        }

    # Find permuted models
    interpolated_model = copy.deepcopy(model_1)
    results = pd.DataFrame()
    for i, model_2 in enumerate(tqdm(model_2_s)):
        lowest_barrier = None
        for _ in tqdm(range(args.iterations), leave=False):
            current_model_p2, barrier = simaneal_find_permuted_model(model_2, model_1, device,
                                                                     partial_train_loader, annealing_params)
            interpolated_model.load_state_dict(interpolate_state_dicts(model_1.state_dict(),
                                               current_model_p2.state_dict()))
            if lowest_barrier is None or barrier < lowest_barrier:
                lowest_barrier = barrier
                model_p2 = copy.deepcopy(current_model_p2)

        interpolated_model.load_state_dict(interpolate_state_dicts(model_1.state_dict(), model_p2.state_dict()))
        train_acc = util.evaluate_model(interpolated_model, device, train_loader, reshape_input=True)["Accuracy"]
        test_acc = util.evaluate_model(interpolated_model, device, test_loader, reshape_input=True)["Accuracy"]
        results = results.append({
            "original_model": model_2.state_dict(),
            "permuted_model": model_p2.state_dict(),
            "barrier": lowest_barrier,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "annealing_parameters": annealing_params,
            "model_1": model_1.state_dict()
        }, ignore_index=True)

    # Save results
    file = args.resultdir + "/lmc_model_pairs{}.pkl".format("_seed_" + str(args.seed) if args.seed else "")
    os.makedirs(args.resultdir, exist_ok=True)
    results.to_pickle(file, protocol=4)

    if args.google_cloud:
        util.upload_blob(google_cloud_bucket, file, file)

    print("done")


if __name__ == '__main__':
    main()
