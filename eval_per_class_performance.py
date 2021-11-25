from typing import Dict
from torch.utils.data.dataloader import DataLoader
import torch
import argparse
import torch.nn as nn
from utils.util import download_folder
from model.builder import build_model
import utils.util as util
import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def eval_per_class_performance(model: torch.nn.Sequential, device: torch.device, data_loader: DataLoader,
                               class_labels: Dict, reshape_input: bool) -> pd.DataFrame:
    target_labels = []
    predicted_labels = []

    # switch to evaluation mode
    model.eval().to(device)
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            if reshape_input is True:
                data, target = data.to(device).view(data.size(0), -1), target.to(device)
            else:
                data, target = data.to(device), target.to(device)

            # compute the output
            output = model(data)
            pred = output.max(1)[1]
            predicted_labels += pred.detach().tolist()
            target_labels += target.detach().tolist()

    precision, recall, f_score, support = precision_recall_fscore_support(target_labels, predicted_labels,
                                                                          labels=list(class_labels.values()))
    eval_result = pd.DataFrame()
    eval_result["label"] = class_labels.values()
    eval_result["class"] = class_labels.keys()
    eval_result["precision"] = precision
    eval_result["recall"] = recall
    eval_result["f1"] = f_score
    eval_result["support"] = support
    return eval_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate per class performance.")
    parser.add_argument('--arch', default='MLP',
                        help='Architecture to eval (options: MLP | resnet | vgg, default: MLP)')
    parser.add_argument('--nlayers', default=1, type=int,
                        help='Number of layers or type of network, e.g. 18 for resnet 18')
    parser.add_argument('--width', required=True, type=int,
                        help='width of the model, for resnet width of first layer, for MLP width of all layers')
    parser.add_argument('--modeldir', type=str, required=True,
                        help='path where the models are saved')
    parser.add_argument('--dataset', type=str, required=True,
                        help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100)')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='Path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--cuda_device_num', default=None, type=int, nargs='+',
                        help='Which cuda devices to use')
    parser.add_argument('--google-cloud', default=False, action='store_true',
                        help='Enables upload/download from google cloud')
    parser.add_argument('--nmodels', type=int, default=5, help='The number of models to use for evalutaion.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for evaluation.')
    parser.add_argument('--nsamples', type=int, default=None,
                        help='Number of samples from each noise type used to evaluate the models.')
    args = parser.parse_args()

    google_cloud_bucket = "lukas_deep"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda and args.cuda_device_num is not None:
        torch.cuda.set_device(args.cuda_device_num[0])

    if args.google_cloud:
        download_folder(google_cloud_bucket, args.modeldir)

    # Load Data and build Model
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

    test_dataset = util.load_data('test', args.dataset, args.datadir, nchannels)
    test_loader = util.get_dataset_loader(test_dataset, args.batch_size, args.nsamples)

    model_files = util.get_model_files_from_dir(args.modeldir, args.nmodels)
    model = build_model(args.arch, nchannels, nclasses, args.width, args.nlayers)

    if args.arch == "MLP_no_flatten":
        reshape_input = True
    else:
        reshape_input = False
        if use_cuda and args.cuda_device_num is not None:
            model = nn.DataParallel(model, device_ids=args.cuda_device_num)
        else:
            model = nn.DataParallel(model)

    # Evaluation
    per_class_measures = pd.DataFrame()
    for model_file in tqdm(model_files):
        model.load_state_dict(torch.load(model_file, map_location=device))
        m_i = eval_per_class_performance(model, device, test_loader, test_dataset.class_to_idx, reshape_input)
        m_i["model"] = model_file
        per_class_measures = per_class_measures.append(m_i)

    result_file = os.path.join(args.modeldir, "per_class_measures.csv")
    per_class_measures.to_csv(result_file)
    if args.google_cloud:
        util.upload_blob(google_cloud_bucket, result_file, result_file)

    print("done")


if __name__ == "__main__":
    main()
