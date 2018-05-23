# Text Recognition Project
This is a simple document reading program that is made to read machine printed text and has a function to detect IBANs.
It was a student project and is far from working perfectly. However, with text that is not blurry the results are ok.
The program mainly consists of two parts, a character extractor and a classifier. Character extraction can either be done by connected component analysis in a binary image made from the input or by extracting maximally stable extremal regions from the input. 
Classification is done by a small convolutional neural network.

## Dependencies
The code is written in python 3.6 and requires the following packages: 

- NumPy
- SciPy
- scikit-learn
- [Pillow](https://pillow.readthedocs.io/en/5.1.x/)
- PyTorch
- [OpenCV](https://pypi.org/project/opencv-python/)
- Matplotlib
- os
- [pyperclip](https://pypi.org/project/pyperclip/)


## Installation
To use the program clone this repository, navigate to it in the terminal and run the interface:

```
$ python interface.py
```

## Documentation
A [documentation](https://lucasmllr.github.io/text_recogition_project/) is available on this repository's GitHub page.

## Report
For a theoretical background on the used algorithms and for results see the [report](https://github.com/lucasmllr/text_recogition_project/raw/master/report.pdf).
For a visualization of the individual steps take a look at this [notebook](https://github.com/lucasmllr/text_recogition_project/blob/master/Evaluate.ipynb).

## Authors
Roman Remme, Lucas-Raphael Mueller, Lucas Moeller

