"""
   CIFAR-10 CIFAR-100, Tiny-ImageNet data loader
"""

import random
import os
import numpy as np
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def fetch_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    # using random crops and horizontal flip for train set
    if params.augmentation:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # ************************************************************************************
    if params.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=False,
                                              download=True, transform=dev_transformer)

    # ************************************************************************************
    elif params.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=False,
                                              download=True, transform=dev_transformer)

    # ************************************************************************************
    elif params.dataset == 'tiny_imagenet':
        data_dir = '/home/souvikku/tiny-imagenet-200/tiny-imagenet-200/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'val/'
        trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
        devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
                                              shuffle=True, num_workers=params.num_workers)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
                                            shuffle=False, num_workers=params.num_workers)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl


def fetch_subset_dataloader(types, params):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """

    # using random crops and horizontal flip for train set
    if params.augmentation:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # ************************************************************************************
    if params.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=False,
                                              download=True, transform=dev_transformer)

    # ************************************************************************************
    elif params.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=False,
                                              download=True, transform=dev_transformer)

    # ************************************************************************************
    elif params.dataset == 'tiny_imagenet':
        data_dir = '/home/souvikku/tiny-imagenet-200/tiny-imagenet-200'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
            }
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                            for x in ['train', 'val']}
        trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=params.batch_size, \
                shuffle=True, num_workers=4, pin_memory=True)
        devloader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=params.batch_size, shuffle=True,\
                num_workers=4, pin_memory=True)

    if params.dataset == 'cifar10' or params.dataset == 'cifar100':
        trainset_size = len(trainset)
        indices = list(range(trainset_size))
        split = int(np.floor(params.subset_percent * trainset_size))
        np.random.seed(230)
        np.random.shuffle(indices)

        train_sampler = SubsetRandomSampler(indices[:split])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
            sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

        devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
            shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl