from typing import List, OrderedDict, Tuple
from model.builder import build_model
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from scipy.sparse import csr_matrix
import argparse
import os
import pickle as pkl
import numpy as np

import utils.util as util


def get_size_of_model(model: torch.nn.Module) -> float:
    torch.save(model.state_dict(), "temp.p")
    mb = os.path.getsize("temp.p")/1e6
    print('Size (MB):', mb)
    os.remove('temp.p')
    return mb


def get_size_of_pkl(object: any) -> float:
    fname = "tmp.pkl"
    with open(fname, "wb") as f:
        pkl.dump(object, f, protocol=4)
    mb = os.path.getsize(fname)/1e6
    print('Size (MB):', mb)
    os.remove(fname)
    return mb


def main():
    parser = argparse.ArgumentParser(
                                     description='Evaluate quantization, and sparse representations.'
    )
    parser.add_argument('--arch', default='MLP',
                        help='Architecture to eval (options: MLP | resnet | vgg, default: MLP)')
    parser.add_argument('--nlayers', default=1, type=int,
                        help='Number of layers or type of network, e.g. 18 for resnet 18')
    parser.add_argument('--width', required=True, type=int,
                        help='width of the model, for resnet width of first layer, for MLP width of all layers')
    parser.add_argument('--modeldir', type=str, required=True,
                        help='path where the models are saved')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset (options: MNIST-C | CIFAR10-C | CIFAR100-C)')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='Path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--cuda_device_num', default=None, type=int, nargs='+',
                        help='Which cuda devices to use')
    parser.add_argument('--seed', default=None, type=int, help="Set a seed for torch.random")
    parser.add_argument('--google-cloud', default=False, action='store_true',
                        help='Enables upload/download from google cloud')
    parser.add_argument('--nmodels', type=int, default=5, help='The number of models to use for evalutaion.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for evaluation.')

    args = parser.parse_args()
    if args.seed is not None:
        print("Setting seed to {}".format(args.seed))
        torch.manual_seed(args.seed)

    google_cloud_bucket = "lukas_deep"
    if args.google_cloud:
        util.download_folder(google_cloud_bucket, args.modeldir)

    device = torch.device("cpu")

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
    test_dataset = util.load_data('val', args.dataset, args.datadir, nchannels)
    train_loader = util.get_dataset_loader(train_dataset, args.batch_size, 5000)
    test_loader = util.get_dataset_loader(test_dataset, args.batch_size, None)

    data_parallel_model = build_model(args.arch, nchannels, nclasses, args.width, args.nlayers)
    data_parallel_model = nn.DataParallel(data_parallel_model)
    data_parallel_model.to(device).eval()

    files = [os.path.join(args.modeldir, f)
             for f in os.listdir(args.modeldir) if os.path.isfile(os.path.join(args.modeldir, f))]
    model_files = [f for f in files if os.path.splitext(f)[1] == ".pt"]
    model_files = model_files[:args.nmodels]

    # Do evaluation
    measures = pd.DataFrame()
    for m_file in tqdm(model_files):
        # Get model
        data_parallel_model.load_state_dict(torch.load(m_file, map_location=device))
        model = list(data_parallel_model.children())[0]  # Get back model from DataParallel wrapper

        # Base measures
        model.to(device).eval()
        acc = util.evaluate_model(model, device, test_loader)["Accuracy"]
        size = get_size_of_pkl(convert_sd(model.state_dict()))
        measures = measures.append({
            "accuracy": acc,
            "size": size,
            "type": "standard",
            "model": m_file
        }, ignore_index=True)

        # create a quantized model instance
        quant_model = build_model(args.arch, nchannels, nclasses, args.width, args.nlayers, quantization=True)
        quant_model = nn.DataParallel(quant_model)
        quant_model.load_state_dict(torch.load(m_file, map_location=device))
        quant_model = list(quant_model.children())[0]
        quant_model.to(device)

        # Prepare, add observers and run to observe
        quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(quant_model, inplace=True)
        _ = util.evaluate_model(quant_model, device, train_loader)

        # Actually convert and evaluate quantizized model
        model_int8 = torch.quantization.convert(quant_model)
        acc_after_quant = util.evaluate_model(model_int8, device, test_loader)["Accuracy"]
        size_after_quant = get_size_of_pkl(convert_sd(model_int8.state_dict()))
        measures = measures.append({
            "accuracy": acc_after_quant,
            "size": size_after_quant,
            "type": "quantization",
            "model": m_file
        }, ignore_index=True)

    # Eval sparse representations of quantizizted model
    original_sd = model_int8.state_dict()
    converted_coo_sd, sparsified_coo_keys = convert_sd(original_sd, sparse_type="coo", sparse_barrier=95)
    compare_dicts(original_sd, undo_convert(converted_coo_sd))  # Check if we did loose any data
    size_sparse_coo = get_size_of_pkl(converted_coo_sd)
    measures = measures.append({
            "accuracy": acc_after_quant,  # Does not change from sparsity
            "size": size_sparse_coo,
            "type": "quantization_coo",
            "sparsified_keys": sparsified_coo_keys,
            "model": m_file
        }, ignore_index=True)

    converted_csr_sd, sparsified_csr_keys = convert_sd(original_sd, sparse_type="csr", sparse_barrier=70)
    compare_dicts(original_sd, undo_convert(converted_csr_sd))  # Check if we did loose any data
    size_sparse_csr = get_size_of_pkl(converted_csr_sd)
    measures = measures.append({
            "accuracy": acc_after_quant,  # Does not change from sparsity
            "size": size_sparse_csr,
            "type": "quantization_csr",
            "sparsified_keys": sparsified_csr_keys,
            "model": m_file
        }, ignore_index=True)

    # Save results
    result_file = os.path.join(args.modeldir, "quantization_measures.csv")
    measures.to_csv(result_file)
    if args.google_cloud:
        util.upload_blob(google_cloud_bucket, result_file, result_file)

    print("done")


def compare_dicts(dict_original: dict, dict_compare: dict, eps=0.000001):
    def compare_element(element_original, element_compare):
        if type(element_original) == torch.Tensor:
            if element_original.is_quantized:
                int_rep = torch.all(element_original.int_repr() == element_compare.int_repr())
                sc = torch.all(
                    torch.abs(element_original.q_per_channel_scales() - element_compare.q_per_channel_scales()) < eps)
                z_p = torch.all(
                    torch.abs(
                        element_original.q_per_channel_zero_points()
                        - element_compare.q_per_channel_zero_points()) < eps
                        )
                return all([int_rep, sc, z_p])
            return torch.all(torch.abs(element_original - element_compare) < eps)
        elif type(element_compare) is tuple:
            return all([compare_element(eo, ec) for eo, ec in zip(element_original, element_compare)])
        else:
            return element_original == element_compare

    assert list(dict_original.keys()) == list(dict_compare.keys())

    for key in dict_original.keys():
        assert compare_element(dict_original[key], dict_compare[key])


def undo_convert(dictionary: dict) -> dict:
    def unconvert_quant(e):
        scales = e['scale']
        zero_points = torch.zeros_like(scales) if e['zero_point'] is None else e['zero_point']
        int_repr = e['int_repr']
        if int_repr['sparse_type'] is None:
            data = torch.tensor(int_repr['array'])
        elif int_repr['sparse_type'] == "csr":
            data = torch.tensor(int_repr['array'].toarray())
        elif int_repr['sparse_type'] == "coo":
            data = int_repr['array'].to_dense()
        else:
            raise NotImplementedError("sparse_type {} not supported".format(int_repr['sparse_type']))

        rep = torch.prod(torch.tensor(data.shape[1:])).item()
        scale_tiled = torch.vstack([scales] * rep).T.view(data.shape)
        zeros_tiled = torch.vstack([zero_points] * rep).T.view(data.shape)

        tensor = data * scale_tiled + zeros_tiled
        quant_tensor = torch.quantize_per_channel(tensor.float(), scales, zero_points, axis=0, dtype=torch.qint8)
        return quant_tensor

    def unconvert_element(element):
        if type(element) == np.ndarray:
            return torch.tensor(element)
        elif type(element) == dict:
            return unconvert_quant(element)
        elif type(element) == tuple:
            return tuple([unconvert_element(e) for e in element])
        elif element is None:
            return None
        elif type(element) is str:
            if element.startswith("torch.dtype;"):
                return eval(element[len("torch.dtype;"):])
            else:
                return element
        else:
            raise NotImplementedError("{}: {} Conversion not implemented".format(type(element), element))

    return {key: unconvert_element(param) for key, param in dictionary.items()}


def convert_sd(model_state_dict: OrderedDict, sparse_type: str = None, sparse_barrier: float = 70) -> Tuple[dict, List]:

    def convert_quant(quant_tensor):
        sparsified = False
        zeros_percent = torch.sum(quant_tensor == 0) / torch.prod(torch.tensor(quant_tensor.shape)) * 100
        if sparse_type is None or (len(quant_tensor.shape) > 2 and sparse_type == "csr")\
           or zeros_percent < sparse_barrier:  # Only sparsify if enough zeros
            int_repr = quant_tensor.int_repr().numpy()
            converted_stype = None
        elif sparse_type == "csr":
            converted_stype = "csr"
            int_repr = csr_matrix(quant_tensor.int_repr().numpy())
            sparsified = True
        elif sparse_type == "coo":
            converted_stype = "coo"
            sparsified = True
            int_repr = quant_tensor.int_repr().to_sparse()  # coo_matrix(quant_tensor.int_repr().numpy())
        else:
            raise NotImplementedError("sparse_type {} not supported".format(sparse_type))

        zero_point = quant_tensor.q_per_channel_zero_points()
        if torch.all(zero_point == 0):
            zero_point = None
        return {
            "zero_point": zero_point,
            "scale": quant_tensor.q_per_channel_scales(),
            "int_repr": {
                "sparse_type": converted_stype,
                "array": int_repr
            }
        }, sparsified

    def convert_element(element):
        if type(element) == torch.Tensor:
            if element.is_quantized:
                return convert_quant(element)
            else:
                return element.detach().numpy(), False
        elif element is None:
            return None, False
        elif type(element) == tuple:
            converted = [convert_element(e) for e in element]
            sparsified = any(c[1] for c in converted)
            return tuple([c[0] for c in converted]), sparsified
        elif type(element) == str:
            return element, False
        elif type(element) == torch.dtype:
            return "torch.dtype;" + str(element), False
        else:
            raise NotImplementedError("{}: {} Conversion not implemented".format(type(element), element))

    converted_dict = {}
    sparsified_keys = []
    for key, param in model_state_dict.items():
        converted_dict[key], sparsified = convert_element(param)
        sparsified_keys.append(key) if sparsified is True else 0

    return converted_dict, sparsified_keys


if __name__ == "__main__":
    main()
