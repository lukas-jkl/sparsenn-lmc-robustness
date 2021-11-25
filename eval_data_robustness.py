import utils.c_data_loader
import torch
import argparse
import torch.nn as nn
from utils.util import download_folder, get_dataset_loader
from model.builder import build_model
import utils.util as util
import os
from tqdm import tqdm
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate performance on corrupted datasets.")
    parser.add_argument('--arch', default='MLP',
                        help='Architecture to eval (options: MLP | resnet | vgg, default: MLP)')
    parser.add_argument('--nlayers', default=1, type=int,
                        help='Number of layers for network or type of network, e.g. 18 for resnet 18')
    parser.add_argument('--width', required=True, type=int,
                        help='Width of the model, for resnet width of first layer, for MLP width of all layers')
    parser.add_argument('--modeldir', type=str, required=True,
                        help='Path where the used models are saved. Will search for files with extension ".pt"')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset (options: MNIST-C | CIFAR10-C | CIFAR100-C)')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='Path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--cuda_device_num', default=None, type=int, nargs='+', help='Which cuda devices to use')
    parser.add_argument('--seed', default=None, type=int, help="Set a seed for torch.random")
    parser.add_argument('--google-cloud', default=False, action='store_true',
                        help='Enables upload/download from google cloud')
    parser.add_argument('--nmodels', type=int, default=3, help='The number of models to use for evalutaion.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for evaluation.')
    parser.add_argument('--severities', type=int, nargs='+', default=[2, 4],
                        help='Severities of corruption used for evaluation.')
    parser.add_argument('--nsamples', type=int, default=1000,
                        help='Number of samples from each noise type used to evaluate the models.')
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
        download_folder(google_cloud_bucket, args.modeldir)

    if args.dataset == 'MNIST-C':
        nchannels = 1
        nclasses = 10
        dataclass = utils.c_data_loader.MNISTC
        corruptions = utils.c_data_loader.mnist_c_corruptions
    elif args.dataset == 'CIFAR10-C':
        nchannels = 3
        nclasses = 10
        dataclass = utils.c_data_loader.CIFAR10C
        corruptions = utils.c_data_loader.cifar_c_corruptions
    elif args.dataset == 'CIFAR100-C':
        nchannels = 3
        nclasses = 100
        dataclass = utils.c_data_loader.CIFAR100C
        corruptions = utils.c_data_loader.cifar_c_corruptions
    else:
        raise Exception("Unknown Dataset")

    model = build_model(args.arch, nchannels, nclasses,
                        args.width, args.nlayers)

    if args.arch == "MLP_no_flatten":
        reshape_input = True
    else:
        reshape_input = False
        if use_cuda and args.cuda_device_num is not None:
            model = nn.DataParallel(model, device_ids=args.cuda_device_num)
        else:
            model = nn.DataParallel(model)

    model_files = util.get_model_files_from_dir(args.modeldir, args.nmodels)

    # Do evaluation
    measures = pd.DataFrame()
    for m_file in tqdm(model_files):
        model.load_state_dict(torch.load(m_file, map_location=device))
        for corruption in tqdm(corruptions.keys(), leave=False):
            if corruption == "labels":
                continue
            for severity in args.severities:
                dataset = dataclass(corruption, severity=severity)
                data_loader = get_dataset_loader(
                    dataset, args.batch_size, args.nsamples)
                eval = util.evaluate_model(
                    model, device, data_loader, None, reshape_input=reshape_input)
                measures = measures.append({"corruption": corruption, "severity": severity, "measure": "accuracy",
                                            "value": eval["Accuracy"], "model": m_file}, ignore_index=True)

    result_file = os.path.join(args.modeldir, "data_robustness_measures.csv")
    measures.to_csv(result_file)
    if args.google_cloud:
        util.upload_blob(google_cloud_bucket, result_file, result_file)

    print("done")


if __name__ == "__main__":
    main()
