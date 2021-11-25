import argparse
import os
import sys
from typing import Dict, Tuple
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import names
import utils.static_sparsify as static_sparsify

import utils.util as util
from model.builder import build_model


def count_weights(model: nn.Module, ltypes=['Linear', 'Conv2d']) -> int:
    count = 0
    for name, child in model.named_children():
        if child._get_name() in ltypes:
            w = child.weight
            count += w.cpu().numel()
            if child.bias is not None:
                b = child.bias
                count += b.cpu().numel()
        else:
            count += count_weights(child)
    return count


def count_zero_weights(model: nn.Module, ltypes=['Linear', 'Conv2d']) -> int:
    count = 0
    for name, child in model.named_children():
        if child._get_name() in ltypes:
            w = child.weight
            count += torch.sum(w.cpu() == 0).item()
            if child.bias is not None:
                b = child.bias
                count += torch.sum(b.cpu() == 0).item()
        else:
            count += count_zero_weights(child)
    return count


def _train(args, model: nn.Sequential, device: torch.device, train_loader: DataLoader,
           criterion, optimizer, epoch: int, static_sparsify_params: Dict, reshape_input=False) -> Tuple[float, float]:
    sum_loss, sum_correct = 0, 0
    # switch to train mode
    model.train()

    for i, (data, target) in enumerate(train_loader):
        if reshape_input is True:
            data, target = data.to(device).view(data.size(0), -1), target.to(device)
        else:
            data, target = data.to(device), target.to(device)

        # compute the output
        output = model(data)
        # compute the classification error and loss
        loss = criterion(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        # compute the gradient and do an SGD step
        optimizer.zero_grad()

        if args.beta_lasso:
            # Lasso L1 regularization
            l1_loss = 0
            with torch.no_grad():
                for _, param in model.named_parameters():
                    l1_loss += args.lambda_fc * torch.norm(param, p=1)

            full_loss = loss + l1_loss
            full_loss.backward()
            optimizer.step()

            # Beta pruning
            with torch.no_grad():
                for _, param in model.named_parameters():
                    mask = param.abs().ge(args.beta * args.lambda_fc)
                    param.mul_(mask)

        else:
            loss.backward()

            if static_sparsify_params is not None:
                # Apply masks to gradients
                for lind, lname in enumerate(static_sparsify_params["lnames_sorted"]):
                    if static_sparsify_params["num_wtf"][lind] > 0:
                        lname_for_statedict = static_sparsify.get_lname_for_statedict(lname)
                        for n, layer in model.named_parameters():
                            if n == "module." + lname_for_statedict or n == lname_for_statedict:
                                mask = torch.clone(static_sparsify_params["smask"][lname])
                                mask = mask.to(device)
                                static_sparsify.mask_tensor(layer.grad, mask, False)
                                del mask
            optimizer.step()

    return (sum_correct / len(train_loader.dataset)) * 100, sum_loss / len(train_loader.dataset)


def validate(args: Dict, model: nn.Sequential, device: torch.device, val_loader: DataLoader, criterion,
             reshape_input=False) -> Tuple[float, float, float, float]:
    accuracy_top1 = util.AverageMeter()
    accuracy_top5 = util.AverageMeter()
    loss = util.AverageMeter()

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if reshape_input is True:
                data, target = data.to(device).view(data.size(0), -1), target.to(device)
            else:
                data, target = data.to(device), target.to(device)

            # compute the output
            output = model(data)

            # compute the classification error and loss
            current_loss = criterion(output, target).item()
            top1, top5 = util.accuracy(output, target, topk=(1, 5))
            accuracy_top1.update(top1, n=len(data))
            accuracy_top5.update(top5, n=len(data))
            loss.update(current_loss, n=len(data))

    return accuracy_top1.avg, accuracy_top5.avg, loss.avg


def run_training(model: nn.Sequential, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                 static_sparsify_params, args,
                 writer: SummaryWriter = None, reshape_input=False) -> Tuple[nn.Sequential, Dict]:

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), args.learningrate, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    print("Starting training on {} samples.".format(len(train_loader.dataset)))
    pbar = tqdm(range(args.epochs), file=sys.stdout)
    for epoch in pbar:
        if epoch != 0 and args.lr_scheduler is True:
            lr_scheduler.step()

        train_acc, train_loss = _train(args, model, device, train_loader, criterion, optimizer, epoch,
                                       static_sparsify_params, reshape_input)
        val_acc, val_acc_top5, val_loss = validate(args, model, device, val_loader, criterion, reshape_input)

        zero_weights = count_zero_weights(model)
        total_weights = count_weights(model)
        remaining_weights = ((total_weights - zero_weights) / total_weights) * 100

        metrics = {
            "loss": train_loss, "accuracy": train_acc, "val_accuracy": val_acc, "val_top5_accuracy": val_acc_top5,
            "val_loss": val_loss, "remaining_weights": remaining_weights, "total_weights": total_weights,
            "zero_weights": zero_weights
        }

        # Write to tensorboard
        if writer is not None:
            for label, value in metrics.items():
                writer.add_scalar(label, value, epoch)
            writer.flush()

        # Print training progress
        desc = "(Loss: {:.6f}, Accuracy: {:.2f}%, Val Accuracy: {:.2f}%, Remaining Weights: {:.2f}%) - ".format(
                train_loss, train_acc, val_acc, remaining_weights)
        pbar.set_description(desc)

        # stop training if the cross-entropy loss is less than the stopping condition
        if train_loss < args.stopcond:
            break

    h_params = {
        "width": args.width, "layers": args.nlayers, "lr": args.learningrate, "momentum": args.momentum,
        "lr_scheduler": args.lr_scheduler, "batchsize": args.batchsize, "weight_decay": args.weight_decay,
        "seed": args.seed, "dataset": args.dataset
        }
    if args.beta_lasso:
        h_params.update({
            "beta": args.beta,
            "lambda": args.lambda_fc
            })
    if static_sparsify_params:
        h_params.update({
                        "static_sparsify_width": args.static_sparsify_width,
                        "static_sparsify_nlayers": args.static_sparsify_nlayers
                        })

    # Write the final results and hparams to tensorboard
    if writer is not None:
        writer.add_hparams(h_params, metrics)
        writer.close()

    print("Training done.")
    return model, metrics, h_params


def main():
    # settings
    parser = argparse.ArgumentParser(description='Training a model (MLP, vgg or ResNet).')
    parser.add_argument('--arch', default='MLP',
                        help='Architecture to train (options: MLP | MLP_no_flatten | ResNet | vgg, default: MLP)')
    parser.add_argument('--nlayers', default=1, type=int,
                        help='Number of layers and type of network, e.g. 18 for ResNet18')
    parser.add_argument('--width', required=True, type=int,
                        help='Width of the model, for ResNet width of first layer, for MLP width of all layers')
    parser.add_argument('--cuda_device_num', default=None, type=int, nargs='+',
                        help='Which cuda devices to use')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='Path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--modeldir', type=str, required=True,
                        help='Path where the model should be saved')
    parser.add_argument('--google-cloud', default=False, action='store_true',
                        help='Enables upload/download from google cloud')
    parser.add_argument('--tensorboard', default=False, action='store_true',
                        help='Enables experiment tracking with tensorboard')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='Name of the dataset (options: MNIST | CIFAR10 | CIFAR100, default: MNIST)')
    parser.add_argument('--seed', default=None, type=int, help="Set a seed for torch.random")
    parser.add_argument('--epochs', default=300, type=int,
                        help='Number of epochs to train (default: 300)')
    parser.add_argument('--stopcond', default=0.01, type=float,
                        help='Stopping condtion based on the cross-entropy loss (default: 0.01)')
    parser.add_argument('--batchsize', default=64, type=int,
                        help='Input batch size (default: 64)')
    parser.add_argument('--learningrate', default=0.001, type=float,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--lr_scheduler', default=False, action='store_true', help='If cosine annealing or fix')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--beta-lasso', default=False, action='store_true',
                        help='Enable training with beta-lasso')
    parser.add_argument('--lambda_fc', default=1e-6, type=float,
                        help='Lambda parameter for beta-lasso')
    parser.add_argument('--beta', default=100, type=float,
                        help='Beta parameter for beta-lasso')
    parser.add_argument('--static_sparsify_width', type=int, default=None,
                        help="Will reduce the parameters to a comparable model with this width")
    parser.add_argument('--static_sparsify_nlayers', type=int, default=None,
                        help="Will reduce the parameters to a comparable model with this depth")

    args = parser.parse_args()
    if args.seed is not None:
        print("Setting seed to {}".format(args.seed))
        torch.manual_seed(args.seed)

    if args.beta_lasso and (args.static_sparsify_nlayers is not None or args.static_sparsify_width is not None):
        raise Exception("Can't use beta-lasso and static-sparsify together")

    google_cloud_bucket = "lukas_deep"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda and args.cuda_device_num is not None:
        torch.cuda.set_device(args.cuda_device_num[0])

    # Load data
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
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
    val_dataset = util.load_data('val', args.dataset, args.datadir, nchannels)
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)

    # Generate subfolder based on args + random name
    if args.tensorboard:
        experiment_name = "neurons_{},lr_{},momentum_{}".format(args.width, args.learningrate, args.momentum)
        if args.beta_lasso:
            experiment_name += ",beta_{},lambda_{}".format(args.beta, args.lambda_fc)
        folder = args.modeldir + "/" + names.get_first_name() + "_" + names.get_last_name() + "_" + experiment_name
    else:
        folder = args.modeldir
    os.makedirs(folder, exist_ok=True)

    writer = SummaryWriter(folder) if args.tensorboard else None

    # Build Model
    model = build_model(args.arch, nchannels, nclasses, args.width, args.nlayers)

    # Static sparsify model
    if args.static_sparsify_width is not None or args.static_sparsify_nlayers is not None:
        target_model_width = args.static_sparsify_width if args.static_sparsify_width is not None else args.width
        target_model_nlayers = args.static_sparsify_nlayers if args.static_sparsify_nlayers is not None \
            else args.nlayers
        target_model = build_model(args.arch, nchannels, nclasses, target_model_width, target_model_nlayers)
        model, static_sparsify_params = static_sparsify.prep_model_and_smask(model, target_model)
    else:
        static_sparsify_params = None

    # Train model
    if args.arch == "MLP_no_flatten":
        reshape_input = True
    else:
        reshape_input = False
        if use_cuda and args.cuda_device_num is not None:
            model = nn.DataParallel(model, device_ids=args.cuda_device_num)
        else:
            model = nn.DataParallel(model)

    model, metrics, h_params = run_training(model, train_loader, val_loader, device, static_sparsify_params, args,
                                            writer, reshape_input)

    # Save model
    model_file = folder + "/model{}_state_dict.pt".format("_seed_" + str(args.seed) if (args.seed is not None) else "")
    torch.save(model.state_dict(), model_file)

    # Save results and h_params
    measures = pd.DataFrame()
    measure_file = folder + "/model{}_measures.csv".format("_seed_" + str(args.seed) if (args.seed is not None) else "")
    metrics.update(h_params)
    measures = measures.append(metrics, ignore_index=True)
    measures.to_csv(measure_file)

    # Upload to google cloud
    if args.google_cloud:
        util.upload_folder(google_cloud_bucket, folder)


if __name__ == '__main__':
    main()
