from arguments import Arguments
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
import os

from model import ConvolutionalNN


class CharDataset(Dataset):
    '''
    subclass of pytorch's Dataset to provide character data to a pytorch model

    Attributes:
        data_tensor (Torch Tensor): holds the images
        target_tensor (Torch Tensor): holds the ground Truth
        label:
    '''

    def __init__(self, data, target, label=None, transform=None):
        self.data_tensor = torch.from_numpy(data).type(torch.FloatTensor)
        self.target_tensor = torch.from_numpy(target).type(torch.LongTensor)
        self.label = label
        self.transform = transform

    def __len__(self):
        return self.target_tensor.size()[0]

    def __getitem__(self, idx):
        '''loads the image, GT tuple with index idx from the data. overrides the respective Dataset method.

        Args:
            idx (int): index in data_tensor and target_tensor

        Returns:
            image, GT
        '''

        data_sample = self.data_tensor[idx]
        if self.transform:
            data_sample = self.transform(data_sample)

        target_sample = self.target_tensor[idx]

        return data_sample, target_sample


def train(epoch, model, data_iterator, criterion, optimizer, args):
    '''
    runs the training for a pytorch model in a single epoch.

    Args:
        epoch: Epoch of training
        model: pytorch model to be trained
        data_iterator: pytorch DataLoader instance
        criterion: instance of a pytorch Loss function
        optimizer: instance of a pytorch Optimizer
        args: Arguments instance
    '''

    model.train()
    for batch_idx, (data, target) in enumerate(data_iterator):
        data, target = Variable(data.view(data.size()[0],1,32,32)), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_iterator.dataset),
                100. * batch_idx / len(data_iterator), loss.data[0]))


def test(model, data_iterator, criterion):
    '''
    evaluated a pytorch model.

    Args:
        model: pytorch model to be tested
        data_iterator: pytorch DataLoader instance
        criterion: instance of a pytorch Optimizer

    Returns:
        test_loss, accuracy
    '''
    model.eval()
    test_loss = 0
    correct = 0
    accu = lambda x: x / len(data_iterator.dataset)
    for data, target in data_iterator:
        data, target = Variable(data.view(data.size()[0], 1, 32, 32)), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(data_iterator.sampler)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_iterator.sampler),
        100. * correct / len(data_iterator.sampler)))
    return test_loss, accu(correct)


def train_and_test(args):
    '''
    runs a training and testing routine for a pytorch model that is created.
    The models state_dict is saved to args.model_path.

    Args:
        args: Arguments instance
    '''

    torch.manual_seed(args.seed)

    # load the dataset
    data = np.load(os.path.join(args.train_path, 'images.npy'))
    target = np.load(os.path.join(args.train_path, 'gt.npy'))

    data_character = CharDataset(data=data, target=target)

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
                              shuffle=True,
                              drop_last=False)

    test_loader = DataLoader(data_character,
                             batch_size=args.batch_size,
                             sampler=test_sampler,
                             # shuffle=True,
                             drop_last=False
                             )

    img_dim = 32 * 32

    model = ConvolutionalNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    for epoch in range(args.epochs):
        if epoch % 3 == 0 and epoch > 0:
            print('reducing lr')
            print()
            for param_group in optimizer.param_groups:
                param_group['lr'] *= .2
        train(epoch, model, train_loader, criterion, optimizer, args)
        test(model, test_loader, criterion)
        # Save results.
        torch.save(model.state_dict(), os.path.join(args.model_path, 'weights.pth'))

if __name__ == '__main__':
    args = Arguments()
    #args.n = 100
    #args.image_path = 'test_data'
    #args.train_path = 'test_data'
    train_and_test(args)
