import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import random

from model.resnet import ResNet18
from trainer import Trainer
from config import get_configs


def get_train_test_trans():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomRotation(5),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_train, transform_test


if __name__ == "__main__":
    seed = 6953
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # get configs
    train_params, dataset_params, callback_params, optimizer_params = get_configs()

    # get model
    model = ResNet18()

    # get data augmentations
    transform_train, transform_test = get_train_test_trans()

    # get trainer
    trainer = Trainer(
        model, 
        dataset_params,
        train_params,
        callback_params,
        optimizer_params,
        transform_train,
        transform_test
    )
    trainer.train()