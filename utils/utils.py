import torch
import torchvision


class DatasetFromSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_cifar10(train_bs=16, test_bs=128, transform_train=None, transform_test=None):
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True)
    
    # train dev split
    train_size = int(0.9 * len(trainset))
    test_size = len(trainset) - train_size
    trainset, devset = torch.utils.data.random_split(trainset, [train_size, test_size])

    trainset = DatasetFromSubset(
        trainset, transform=transform_train
    )
    devset = DatasetFromSubset(
        devset, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_bs, shuffle=True)
    devloader = torch.utils.data.DataLoader(
        devset, batch_size=test_bs, shuffle=False)

    # load testset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_bs, shuffle=False)

    return trainloader, devloader, testloader

