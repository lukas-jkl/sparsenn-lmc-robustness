import io
import os
import pickle as pkl
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from google.cloud import storage
import google


def upload_blob(bucket_name: str, source_file_name: str, destination_blob_name: str,
                blob_path_prefix: str = "deep_ensemble_master_thesis/"):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path_prefix + destination_blob_name)
    blob.upload_from_filename(source_file_name, timeout=480)


def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: str,
                  blob_path_prefix: str = "deep_ensemble_master_thesis/"):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path_prefix + source_blob_name)
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    try:
        blob.download_to_filename(destination_file_name)
    except google.api_core.exceptions.NotFound as e:
        os.remove(destination_file_name)
        print(e)
        raise FileNotFoundError


def upload_pkl(bucket_name: str, pickle_out: str, destination_blob_name: str,
               blob_path_prefix: str = "deep_ensemble_master_thesis/"):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path_prefix + destination_blob_name)
    blob.upload_from_string(pickle_out)


def download_pkl(bucket_name: str, source_blob_name: str, blob_path_prefix: str = "deep_ensemble_master_thesis/"):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path_prefix + source_blob_name)
    pickle_in = blob.download_as_string()
    return pkl.loads(pickle_in)


def download_blob_if_not_exists(bucket_name: str, source_blob_name: str, destination_file_name: str,
                                blob_path_prefix: str = "deep_ensemble_master_thesis/"):
    if not os.path.isfile(destination_file_name):
        download_blob(bucket_name, source_blob_name, destination_file_name, blob_path_prefix)


def upload_folder(bucket_name: str, folder: str, blob_path_prefix: str = "deep_ensemble_master_thesis/"):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    subdirs = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    for file in files:
        print("uploading file:", file)
        upload_blob(bucket_name, folder + "/" + file, folder + "/" + file, blob_path_prefix)
    for dir in subdirs:
        print("uploading subdir:", dir)
        upload_folder(bucket_name,  folder + "/" + dir, blob_path_prefix)
    print("done")


def download_folder(bucket_name: str, folder: str, blob_path_prefix: str = "deep_ensemble_master_thesis/"):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=blob_path_prefix + folder)  # Get list of files
    print("Starting download of", folder)
    for blob in blobs:
        if blob.name[-1] == "/":
            # skip folders
            continue
        print("loading: ", blob.name)
        name_without_prefix = blob.name[len(blob_path_prefix):]
        download_blob_if_not_exists(bucket_name, name_without_prefix, name_without_prefix, blob_path_prefix)
    print("done")


def load_data(split: str, dataset_name: str, datadir: str, nchannels: int) -> torch.utils.data.Dataset:
    # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    if dataset_name == 'MNIST':
        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
    elif dataset_name == 'SVHN':
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
    elif dataset_name == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    elif dataset_name == 'CIFAR100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

    tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    get_dataset = getattr(datasets, dataset_name)
    if dataset_name == 'SVHN':
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
    else:
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)

    return dataset


def get_dataset_loader(dataset: torch.utils.data.Dataset, batch_size: int, samples: int = None) -> DataLoader:
    if samples:
        data_indices = np.array(range(len(dataset)))
        np.random.shuffle(data_indices)
        dataset = torch.utils.data.Subset(dataset, data_indices[:samples])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def unpack_dataloader(loader: DataLoader, device: torch.device, reshape_input: bool = False) -> Tuple[List, List]:
    inputs, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            if reshape_input is True:
                data, target = data.to(device).view(data.size(0), -1), target.to(device)
            else:
                target, data = target.to(device), data.to(device)
            inputs.append(data)
            targets.append(target)
    return inputs, targets


def get_noisy_data_loader(original_loader: DataLoader, device: torch.device, noise_mean: float, noise_std_dev: float) \
                          -> DataLoader:
    data, targets = unpack_dataloader(original_loader, device)
    data = torch.vstack(data)
    targets = torch.hstack(targets)
    data.add_(torch.normal(mean=noise_mean, std=noise_std_dev, size=data.shape).to(device))
    noisy_dataset = torch.utils.data.TensorDataset(data, targets)
    return DataLoader(noisy_dataset, batch_size=original_loader.batch_size)


def apply_noise_to_model(model: nn.Sequential,  std_dev: float,
                         noise_type: str = "model_weights_abs", ltypes=['Linear', 'Conv2d']):
    for lname, child in model.named_children():
        ltype = child._get_name()
        if ltype in ltypes:
            with torch.no_grad():
                if noise_type == "model_weights":
                    child.weight.add_(torch.normal(mean=0, std=std_dev, size=child.weight.shape) * child.weight)
                    if child.bias is not None:
                        child.bias.add_(torch.normal(mean=0, std=std_dev, size=child.bias.shape) * child.bias)
                elif noise_type == "model_weights_2":
                    child.weight.add_(torch.normal(mean=0, std=std_dev * torch.square(child.weight)))
                    if child.bias is not None:
                        child.bias.add_(torch.normal(mean=0, std=std_dev * torch.square(child.bias)))
                elif noise_type == "model_weights_abs":
                    child.weight.add_(torch.normal(mean=0, std=std_dev * torch.abs(child.weight)))
                    if child.bias is not None:
                        child.bias.add_(torch.normal(mean=0, std=std_dev * torch.abs(child.bias)))
                elif noise_type == "model_weights_norm":
                    child.weight.add_(torch.normal(mean=0, std=std_dev *
                                      (torch.ones_like(child.weight) * torch.norm(child.weight, p="fro"))))
                    if child.bias is not None:
                        child.bias.add_(torch.normal(mean=0, std=std_dev *
                                        (torch.ones_like(child.bias) * torch.norm(child.bias, p="fro"))))
                else:
                    raise NotImplementedError("Noise type not implemented")
        else:
            apply_noise_to_model(child, std_dev, noise_type, ltypes)


def load_gpu_pkl(file: str, device: torch.device) -> Any:
    # Workaround for unpickle on cpu-only machine https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
    class Unpickler(pkl.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location=device)
            else:
                return super().find_class(module, name)

    with open(file, 'rb') as f:
        loaded = Unpickler(f).load()

    return loaded


def evaluate_model(model: nn.Sequential, device: torch.device, data_loader: DataLoader, criterion=None,
                   reshape_input: bool = False) -> Dict[str, float]:
    sum_loss, sum_correct = 0, 0

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

            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            if criterion:
                sum_loss += len(data) * criterion(output, target).item()

    accuracy = sum_correct / len(data_loader.dataset)
    loss = sum_loss / len(data_loader.dataset)

    if criterion:
        return {
            "Accuracy": accuracy * 100,
            "Loss": loss
        }
    else:
        return {
            "Accuracy": accuracy * 100
        }


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)) -> List[float]:
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(0)
        res.append(torch.mul(correct_k, 100. / batch_size).item())
    return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cached_function_call(cache_file: str, function: Callable, *args, **kwargs) -> Any:
    bucket = "lukas_deep"
    try:
        # Try to load file from local disc
        with open(cache_file, "rb") as f:
            return pkl.load(f)
    except FileNotFoundError:
        try:
            # Try to download
            download_blob(bucket, cache_file, cache_file)
            with open(cache_file, "rb") as f:
                return pkl.load(f)
        except FileNotFoundError:
            # Calculate
            result = function(*args, **kwargs)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            # Store to local disc
            with open(cache_file, "wb") as f:
                pkl.dump(result, f, protocol=4)

            # Upload
            upload_blob(bucket, cache_file, cache_file)
            return result


def get_model_files_from_dir(modeldir: str, nmodels: int, extension: str = ".pt") -> List[str]:
    files = [os.path.join(modeldir, f)
             for f in os.listdir(modeldir) if os.path.isfile(os.path.join(modeldir, f))]
    model_files = [f for f in files if os.path.splitext(f)[1] == extension]
    model_files = model_files[:nmodels]
    return model_files
