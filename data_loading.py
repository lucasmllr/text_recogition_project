import numpy as np
import pylab as plt
import os
import csv
from torch.utils.data import DataLoader, Dataset
import processing
from blob_extraction import find_blobs


class ImageDataset(Dataset):
    '''
    subclass of pytorch's Dataset to provide extracted characters to a model.
    '''

    def __init__(self, path='data', size=28):
        '''
        initiaizes an ImagaDataset instance

        Args:
            path (string): path of dataset. must contain jpg images and a file named truth.txt
            size (int): size of character images handed to the model
        '''

        # images
        self.path = path
        self.raws = self.load_raws()
        self.truth = self.load_truth()

        # characters
        self.size = size
        self.blobs = []
        self.blob_truth = []
        self.extract_blobs()


    def load_raws(self):
        '''
        loads raw images from the data directory and stores it as ndarrays

        Returns:
            list of ndarrays
        '''

        raws = []

        for file in os.listdir(self.path):
            if file.endswith('.jpg'):
                file_path = os.path.join(self.path, file)
                raws.append(processing.load_img(file_path))

        return raws


    def load_truth(self):
        '''
        loads ground truth strings from "truth.txt" file

        Returns:
            a list of strings
        '''

        truth_path = os.path.join(self.path, 'truth.txt')
        if not os.path.exists(truth_path):
            raise IOError('There is no file "truth.txt" in {}'.format(self.path))

        truth = []
        with open(truth_path) as input:
            for line in csv.reader(input):
                truth.append(line[0])

        return truth


    def extract_blobs(self):
        '''
        function to perform blob extraction on raw images.
        Data is only used if blob extraction finds the right number of characters in an image.
        '''

        for truth, img in zip(self.truth, self.raws):
            chars = find_blobs(img)

            if len(chars) == len(truth):
                for j in range(len(chars)):
                    char = processing.rescale(chars[j], self.size)
                    self.blobs.append(char)
                    self.blob_truth.append(truth[j])

        return


    def __len__(self):
        ''' overrides funciton of Dataset class'''
        return len(self.blobs)


    def __getitem__(self, idx):
        ''' overrides function of Dataset class'''
        return self.blobs[idx], self.blob_truth[idx]


if __name__ == "__main__":

    imgs = ImageDataset()
    for i in range(10):
        char, label = imgs.__getitem__(idx=i)
        plt.imshow(char)
        plt.title(label)
        plt.show()