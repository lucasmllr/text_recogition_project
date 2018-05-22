# Text Recognition Project
This is a simple document reading program that is made to read machine printed text and has a function to detect IBANs.
It was a student project and is far from working perfectly. However, with text that is not blurry the results are ok.
The program mainly consists of two parts, a character extractor and a classifier. Character extraction can either be done by connected component analysis in a binary image made from the input or by extracting maximally stable extremal regions from the input. 
Classification is done by a small convolutional neural network.

## Dependencies
The code is written in python 3.6 and requires the following packages: 
numpy, scipy, sklearn, pillow, pytorch, opencv, matplotlib, os, pyperclip

## Installation
To use the program clone this repository, navigate to it in the terminal and run the interface:

´´´sh
$ python interface.py
´´´

## Documentation
A documentation is available on this repositorie's GitHub page [here](https://lucasmllr.github.io/text_recogition_project/)

