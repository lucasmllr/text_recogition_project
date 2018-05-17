# Text Recognition Project
This is a simple document reading program that is made to read machine printed text.
It mainly consists of two parts, a character extractor and a classifier. Character extraction can either be done by connected component analysis in a binary image made from the input are by extracting maximally stable extremal regions from the inout. 
Classification is done by a small convolutional neural network.

## Dependencies
The code is written in python 3.6 and requires the following packages: 
numpy, scipy, sklearn, pillow, pytorch, , opencv, matplotlib, os, pyperclip

## Installation
To use the program clone this repository, navigate to it in the terminal and start the interface with the following cammand:

´´´
python interface.py
´´´

## Documentation
A documentation is available on this repositories GitHub page:

