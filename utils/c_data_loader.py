import os
import wget
import tarfile
import zipfile
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import PIL

cifar_c_corruptions = {
    "brightness": "brightness.npy",
    "contrast": "contrast.npy",
    "defocus_blur": "defocus_blur.npy",
    "elastic_transform": "elastic_transform.npy",
    "fog": "fog.npy",
    "frost": "frost.npy",
    "gaussian_blur": "gaussian_blur.npy",
    "gaussian_noise": "gaussian_noise.npy",
    "glass_blur": "glass_blur.npy",
    "impulse_noise": "impulse_noise.npy",
    "jpeg_compression": "jpeg_compression.npy",
    "labels": "labels.npy",
    "motion_blur": "motion_blur.npy",
    "pixelate": "pixelate.npy",
    "saturate": "saturate.npy",
    "shot_noise": "shot_noise.npy",
    "snow": "snow.npy",
    "spatter": "spatter.npy",
    "speckle_noise": "speckle_noise.npy",
    "zoom_blur": "zoom_blur.npy"
}

mnist_c_corruptions = {
    "brightness": "brightness",
    "canny_edges": "canny_edges",
    "dotted_line": "dotted_line",
    "fog": "fog",
    "glass_blur": "glass_blur",
    "identity": "identity",
    "impulse_noise": "impulse_noise",
    "motion_blur": "motion_blur",
    "rotate": "rotate",
    "scale": "scale",
    "shear": "shear",
    "shot_noise": "shot_noise",
    "spatter": "spatter",
    "stripe": "stripe",
    "translate": "translate",
    "zigzag": "zigzag"
}


def _download_and_unpack_dataset(datadir, file, url, unpackfolder):
    outfile = os.path.join(datadir, file)
    if not os.path.isfile(outfile):
        wget.download(url, out=outfile)
    if not os.path.isdir(os.path.join(datadir, unpackfolder)):
        extension = os.path.splitext(file)[1]
        if "tar" in extension:
            with tarfile.open(outfile) as tar:
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), desc="Unpacking - "):
                    tar.extract(member, path=datadir)
        elif "zip" in extension:
            with zipfile.ZipFile(outfile) as zip:
                zip.extractall(path=datadir)
        else:
            raise Exception("Can't unpack file ", file)


class CIFAR10C(Dataset):
    def __init__(self, corruption, datadir="datasets", severity=2, transform=None, target_transform=None):
        unpackfolder = "CIFAR-10-C"
        _download_and_unpack_dataset(datadir, "CIFAR-10-C.tar",
                                     "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar", unpackfolder)
        datafile = os.path.join(datadir, unpackfolder, cifar_c_corruptions[corruption])
        self.data = np.load(datafile)[(severity-1)*10000:severity*10000]
        labelsfile = os.path.join(datadir, unpackfolder, "labels.npy")
        self.labels = torch.tensor(np.load(labelsfile)[(severity-1)*10000:severity*10000], dtype=torch.long)
        if transform is None:
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = PIL.Image.fromarray(image)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CIFAR100C(Dataset):
    def __init__(self, corruption, datadir="datasets", severity=2, transform=None, target_transform=None):
        unpackfolder = "CIFAR-100-C"
        _download_and_unpack_dataset(datadir, "CIFAR-100-C.tar",
                                     "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar", unpackfolder)
        datafile = os.path.join(datadir, unpackfolder, cifar_c_corruptions[corruption])
        self.data = np.load(datafile)[(severity-1)*10000:severity*10000]
        labelsfile = os.path.join(datadir, unpackfolder, "labels.npy")
        self.labels = torch.tensor(np.load(labelsfile)[(severity-1)*10000:severity*10000], dtype=torch.long)
        if transform is None:
            normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
            transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = PIL.Image.fromarray(image)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class MNISTC(Dataset):
    def __init__(self, corruption, datadir="datasets", severity=1, transform=None, target_transform=None):
        if severity != 1:
            raise Exception("MNISTC does only have severity 1")
        unpackfolder = "mnist_c"
        _download_and_unpack_dataset(datadir, "mnist_c.zip",
                                     "https://zenodo.org/record/3239543/files/mnist_c.zip", unpackfolder)
        datafile = os.path.join(datadir, unpackfolder, mnist_c_corruptions[corruption], "test_images.npy")
        self.data = np.load(datafile)
        labelfile = os.path.join(datadir, unpackfolder, mnist_c_corruptions[corruption], "test_labels.npy")
        self.labels = torch.tensor(np.load(labelfile), dtype=torch.long)
        if transform is None:
            normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
            transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx][:, :, 0]
        image = PIL.Image.fromarray(image)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
