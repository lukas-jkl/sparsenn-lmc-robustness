import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd
import os
import copy
from tqdm import tqdm
import utils.util as util
from model.builder import build_model
import permute


def noisy_evaluation(model: nn.Sequential, device: torch.device, data_loader: DataLoader, criterion, nsamples: int = 16,
                     noise_types=["noisy_data", "noisy_model"], reshape_input: bool = False):
    generalization_measures = pd.DataFrame()
    std_devs = [0, 0.25, 0.5, 0.75, 1, 2]
    for noise_type in noise_types:
        for std_dev in tqdm(std_devs, desc="Evaluate generalization performance - Progress:", leave=False):
            for _ in range(nsamples):
                eval_loader = data_loader
                if std_dev != 0:
                    if noise_type == "noisy_data":
                        eval_model = copy.deepcopy(model)
                        eval_loader = util.get_noisy_data_loader(data_loader, device, 0, std_dev)
                    elif noise_type == "noisy_model":
                        eval_model = copy.deepcopy(model)
                        util.apply_noise_to_model(eval_model, std_dev, "model_weights_abs")
                else:
                    eval_model = copy.deepcopy(model)

                measures = util.evaluate_model(eval_model, device, eval_loader, criterion, reshape_input)
                del eval_model
                generalization_measures = generalization_measures.append({
                    "noise_type": noise_type, "std_dev": std_dev, "measure": "accuracy", "value": measures["Accuracy"]
                }, ignore_index=True)
                generalization_measures = generalization_measures.append({
                    "noise_type": noise_type, "std_dev": std_dev, "measure": "loss", "value": measures["Loss"]
                }, ignore_index=True)

    return generalization_measures


def barrier_evaluation(m_1: nn.Sequential, m_1_file: str, m_2: nn.Sequential, m_2_file: str, device: torch.device,
                       data_loader: DataLoader, criterion, reshape_input: bool = False):

    m_1_measures = util.evaluate_model(m_1, device, data_loader, criterion, reshape_input)
    m_1.cpu()  # Move back to cpu to preserve memory

    m_2_measures = util.evaluate_model(m_2, device, data_loader, criterion, reshape_input)
    m_2.cpu()  # Move back to cpu to preserve memory

    m_1.load_state_dict(permute.interpolate_state_dicts(m_1.state_dict(), m_2.state_dict()))
    barrier_measures = util.evaluate_model(m_1, device, data_loader, criterion, reshape_input)
    m_1.cpu()  # Move back to cpu to preserve memory

    result = {
        "m1": m_1_file,
        "m2": m_2_file,
        "m1_accuracy": m_1_measures["Accuracy"],
        "m2_accuracy": m_2_measures["Accuracy"],
        "avg_accuracy": (m_1_measures["Accuracy"] + m_2_measures["Accuracy"]) / 2.,
        "barrier_accuracy": barrier_measures["Accuracy"],
        "m1_loss": m_1_measures["Loss"],
        "m2_loss": m_2_measures["Loss"],
        "avg_loss": (m_1_measures["Loss"] + m_2_measures["Loss"]) / 2.,
        "barrier_loss": barrier_measures["Loss"]
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generalization of models, estimate barrier and noisy performance")
    parser.add_argument('--arch', default='MLP',
                        help='Architecture to eval (options: MLP | resnet | vgg, default: MLP)')
    parser.add_argument('--nlayers', default=1, type=int,
                        help='Number of layers or type of network, e.g. 18 for resnet 18')
    parser.add_argument('--width', required=True, type=int,
                        help='width of the model, for resnet width of first layer, for MLP width of all layers')
    parser.add_argument('--modeldir', type=str, required=True,
                        help='path where the models are saved')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100, default: MNIST)')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='Path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--cuda_device_num', default=None, type=int, nargs='+',
                        help='Which cuda devices to use')
    parser.add_argument('--seed', default=None, type=int, help="Set a seed for torch.random")
    parser.add_argument('--google-cloud', default=False, action='store_true',
                        help='Enables upload/download from google cloud')
    parser.add_argument('--barrier-evaluation', default=False,
                        action='store_true', help='Also evaluate barrier.')
    parser.add_argument('--nmodels', type=int, default=3,
                        help='The number of models to use for evalutaion.')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for evaluation.')

    args = parser.parse_args()
    if args.seed is not None:
        print("Setting seed to {}".format(args.seed))
        torch.manual_seed(args.seed)

    google_cloud_bucket = "lukas_deep"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    unpack_device = torch.device("cpu")
    if use_cuda and args.cuda_device_num is not None:
        torch.cuda.set_device(args.cuda_device_num[0])

    if args.google_cloud:
        util.download_folder(google_cloud_bucket, args.modeldir)

    # Load Data and Models
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
    criterion = nn.CrossEntropyLoss().to(device)

    model_files = util.get_model_files_from_dir(args.modeldir, args.nmodels)

    test_barrier_measures_frame = pd.DataFrame()
    train_barrier_measures_frame = pd.DataFrame()
    noisy_measures_frame = pd.DataFrame()

    m_1 = build_model(args.arch, nchannels, nclasses, args.width, args.nlayers)
    if args.arch == "MLP_no_flatten":
        reshape_input = True
    else:
        reshape_input = False
        if use_cuda and args.cuda_device_num is not None:
            m_1 = nn.DataParallel(m_1, device_ids=args.cuda_device_num)
        else:
            m_1 = nn.DataParallel(m_1)

    # Noisy weight evaluation
    for m_1_file in tqdm(model_files):
        test_loader = util.get_dataset_loader(test_dataset, args.batch_size, None)
        m_1.load_state_dict(torch.load(m_1_file, map_location=unpack_device))
        noisy_measures = noisy_evaluation(m_1, device, test_loader, criterion, 5, ["noisy_model"], reshape_input)
        noisy_measures["model"] = m_1_file
        noisy_measures_frame = noisy_measures_frame.append(noisy_measures)

    noisy_results_file = os.path.join(args.modeldir, "noisy_abs_weight_measures.csv")
    noisy_measures_frame.to_csv(noisy_results_file)
    if args.google_cloud:
        util.upload_blob(google_cloud_bucket, noisy_results_file, noisy_results_file)

    # Additionally perform barrier evaluation if requested
    if args.barrier_evaluation:
        m_2 = copy.deepcopy(m_1)
        for i in tqdm(range(0, len(model_files) - 1, 2)):
            train_loader = util.get_dataset_loader(train_dataset, args.batch_size, 6000)  # different samples each iter
            test_loader = util.get_dataset_loader(test_dataset, args.batch_size, 1000)  # different samples each iter
            m_1_file = model_files[i]
            m_2_file = model_files[i + 1]
            m_1.load_state_dict(torch.load(m_1_file, map_location=unpack_device))
            m_2.load_state_dict(torch.load(m_2_file, map_location=unpack_device))

            # Test Barrier evaluation
            test_barrier_results = barrier_evaluation(
                m_1, m_1_file, m_2, m_2_file, device, test_loader, criterion, reshape_input)
            m_1.cpu()  # Move back to cpu to preserve memory
            test_barrier_measures_frame = test_barrier_measures_frame.append(test_barrier_results, ignore_index=True)

            # Train Barrier evaluation
            train_barrier_results = barrier_evaluation(
                m_1, m_1_file, m_2, m_2_file, device, train_loader, criterion, reshape_input)
            m_1.cpu()  # Move back to cpu to preserve memory
            train_barrier_measures_frame = train_barrier_measures_frame.append(
                train_barrier_results, ignore_index=True)

        test_barrier_result_file = os.path.join(args.modeldir, "barrier_measures.csv")
        test_barrier_measures_frame.to_csv(test_barrier_result_file)
        if args.google_cloud:
            util.upload_blob(google_cloud_bucket, test_barrier_result_file, test_barrier_result_file)

        train_barrier_result_file = os.path.join(args.modeldir, "train_barrier_measures.csv")
        train_barrier_measures_frame.to_csv(train_barrier_result_file)
        if args.google_cloud:
            util.upload_blob(google_cloud_bucket, train_barrier_result_file, train_barrier_result_file)

    print("done")


if __name__ == "__main__":
    main()
