import argparse
import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, utils
from torch.autograd import Variable

from model import ConvolutionalNN


class Args:
    def __init__(self):
        self.batch_size = 300
        self.epochs = 50
        self.lr = 0.001
        self.momentum = 0.5
        self.test_size = 0.1
        self.seed = np.random.randint(32000)
        self.log_interval = 100
        self.shuffle = True


class Chardata(Dataset):
    def __init__(self, data, target, label=None, transform=None):
        self.data_tensor = torch.from_numpy(data).type(torch.FloatTensor)
        self.target_tensor = torch.from_numpy(target).type(torch.LongTensor)
        self.label = label
        self.transform = transform

    def __len__(self):
        return self.target_tensor.shape[0]

    def __getitem__(self, idx):

        data_sample = self.data_tensor[idx]
        if self.transform:
            data_sample = self.transform(sample)

        target_sample = self.target_tensor[idx]

        return data_sample, target_sample


def train(epoch, data_iterator, criterion, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(data_iterator):
        data, target = Variable(data.view(data.shape[0],1,32,32)), Variable(target)
        optimizer.zero_grad
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_iterator.dataset),
                100. * batch_idx / len(data_iterator), loss.data[0]))


def test(data_iterator, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    accu = lambda x: x / len(data_iterator.dataset)
    for data, target in data_iterator:
        data, target = Variable(data.view(data.shape[0], 1, 32, 32)), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.sampler)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_iterator.sampler),
        100. * correct / len(data_iterator.sampler)))
    return test_loss, accu(correct)


if __name__ == '__main__':
    args = Args()
    torch.manual_seed(args.seed)

    # load the dataset
    data = np.load('data/data.npy')
    target = np.load('data/target.npy')

    data_character = Chardata(data=data, target=target)

    num_samples = len(data_character)
    indices = list(range(num_samples))
    split = int(np.floor(args.test_size * num_samples))

    if args.shuffle:
        random_seed = 5
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(data_character,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              # shuffle=True,
                              drop_last=True)

    test_loader = DataLoader(data_character,
                             batch_size=args.batch_size,
                             sampler=test_sampler,
                             # shuffle=True,
                             drop_last=True
                             )

    img_dim = 32 * 32

    model = ConvolutionalNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    for epoch in range(args.epochs):
        train(epoch, train_loader, criterion, optimizer)
        test(test_loader, criterion)
        # Save results.
        torch.save(model.state_dict(), 'data/ocr_torchdict.pth')
