import argparse
import os
from typing import Dict
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import torch
import torch.nn.utils.prune as prune
import utils.static_sparsify as static_sparsify

from model.builder import build_model
import utils.util as util
import train_model


def magnitude_pruning(model: nn.Sequential, prune_ratio: float, ltypes=['Linear', 'Conv2d']):
    for lname, child in model.named_children():
        ltype = child._get_name()
        if ltype in ltypes:
            with torch.no_grad():
                prune.l1_unstructured(child, name="weight", amount=prune_ratio)
                prune.remove(child, "weight")
        else:
            magnitude_pruning(child, prune_ratio, ltypes)


def calculate_metrics(model: nn.Sequential, train_loader: DataLoader, test_loader: DataLoader, device: torch.device,
                      reshape_input: bool = True) -> Dict:
    train_eval = util.evaluate_model(model, device, train_loader, reshape_input=reshape_input)
    test_eval = util.evaluate_model(model, device, test_loader, reshape_input=reshape_input)
    zero_weights = train_model.count_zero_weights(model)
    total_weights = train_model.count_weights(model)
    return {
        "accuracy": train_eval["Accuracy"],
        "val_accuracy": test_eval["Accuracy"],
        "total_weights": total_weights,
        "zero_weights": zero_weights
    }


def main():
    parser = argparse.ArgumentParser(description='Perform magnitude pruning on trained models in a directory.')
    parser.add_argument('--arch', default='MLP',
                        help='Architecture to train (options: MLP | resnet | vgg, default: MLP)')
    parser.add_argument('--nlayers', default=1, type=int,
                        help='Number of layers or type of network, e.g. 18 for resnet 18')
    parser.add_argument('--width', required=True, type=int,
                        help='Width of the model, for resnet width of first layer, for MLP width of all layers')
    parser.add_argument('--cuda_device_num', default=None, type=int, nargs='+',
                        help='Which cuda devices to use')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='Path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--modeldir', type=str, required=True,
                        help='Path where the model should be saved')
    parser.add_argument('--google-cloud', default=False, action='store_true',
                        help='Enables upload/download from google cloud')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='Name of the dataset (options: MNIST | CIFAR10 | CIFAR100, default: MNIST)')
    parser.add_argument('--seed', default=None, type=int, help="Set a seed for torch.random")
    parser.add_argument('--prune_target_width', type=int, default=None,
                        help="Will reduce the parameters to a comparable model with this width")
    parser.add_argument('--prune_target_nlayers', type=int, default=None,
                        help="Will reduce the parameters to a comparable model with this depth")
    parser.add_argument('--num_models', default=5)

    args = parser.parse_args()
    if args.seed is not None:
        print("Setting seed to {}".format(args.seed))
        torch.manual_seed(args.seed)

    google_cloud_bucket = "lukas_deep"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda and args.cuda_device_num is not None:
        torch.cuda.set_device(args.cuda_device_num[0])

    if args.google_cloud:
        util.download_folder(google_cloud_bucket, args.modeldir)

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

    train_dataset = util.load_data('train', args.dataset, args.datadir, nchannels)
    test_dataset = util.load_data('test', args.dataset, args.datadir, nchannels)

    # check target model
    target_model_width = args.prune_target_width if args.prune_target_width is not None else args.width
    target_model_nlayers = args.prune_target_nlayers if args.prune_target_nlayers is not None else args.nlayers
    target_model = build_model(args.arch, nchannels, nclasses, target_model_width, target_model_nlayers)
    num_W_tot_target = static_sparsify.get_num_W_tot(target_model)
    del target_model

    # Load existing model
    files = [os.path.join(args.modeldir, f)
             for f in os.listdir(args.modeldir) if os.path.isfile(os.path.join(args.modeldir, f))]
    model_files = [f for f in files if os.path.splitext(f)[1] == ".pt"]
    model_files = model_files[:args.num_models]
    model = build_model(args.arch, nchannels, nclasses, args.width, args.nlayers)
    num_W_model = static_sparsify.get_num_W_tot(model)

    prune_ratio = 1 - (num_W_tot_target / num_W_model)

    if args.arch == "MLP_no_flatten":
        reshape_input = True
    else:
        reshape_input = False
        if use_cuda and args.cuda_device_num is not None:
            model = nn.DataParallel(model, device_ids=args.cuda_device_num)
        else:
            model = nn.DataParallel(model)

    parent_dir = os.path.dirname(args.modeldir)
    savedir = os.path.join(parent_dir, "post_train_sparsify_{}".format(args.prune_target_width))
    os.makedirs(savedir, exist_ok=True)

    for model_file in tqdm(model_files):
        measures_file = model_file.replace("state_dict.pt", "measures.csv")
        model.load_state_dict(torch.load(model_file, map_location=device))

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True, **kwargs)

        # Metrics before pruning
        m = calculate_metrics(model, train_loader, test_loader, device, reshape_input)
        metrics = {key + "_before_prune": value for key, value in m.items()}

        # Prune
        magnitude_pruning(model, prune_ratio)

        # Metrics after pruning
        m = calculate_metrics(model, train_loader, test_loader, device, reshape_input)
        metrics.update(m)

        # Save results
        metrics_frame = pd.DataFrame()
        metrics_frame = metrics_frame.append(metrics, ignore_index=True)
        pruned_model_measures_file = os.path.join(savedir, os.path.basename(measures_file))
        metrics_frame.to_csv(pruned_model_measures_file)

        # Save model
        pruned_model_file = os.path.join(savedir, os.path.basename(model_file))
        torch.save(model.state_dict(), pruned_model_file)

        if args.google_cloud:
            util.upload_blob(google_cloud_bucket, pruned_model_file, pruned_model_file)
            util.upload_blob(google_cloud_bucket, pruned_model_measures_file, pruned_model_measures_file)

    print("done")


if __name__ == '__main__':
    main()
