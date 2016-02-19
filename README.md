
# Cross-dimensional Weighting (CroW) aggregation

This repository contains a Python implementation of CroW aggregation for deep convolution features. CroW is an efficient non-parametric weighting and aggregation scheme to transform convolutional image features to a compact global image feature. This repository contains code to evaluate these global CroW features for image retrieval on common retrieval benchmarks. A full description of this work can be found at [http://arxiv.org/abs/1512.04065](http://arxiv.org/abs/1512.04065).

### Installation

This repository contains scripts to download the [Oxford](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) and [Paris](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) image retrieval benchmark datasets, as well as the VGG16 pre-trained convolutional neural network model for feature extraction. The Python scripts in this repository offer general utilities for extracting features from [Caffe](http://caffe.berkeleyvision.org/) models, experimenting with different feature aggregation schemes, and evaluating retrieval performance on the Oxford and Paris datasets. The model and dataset requirements are only necessary to run evaluation.

#### Requirements

For feature extraction, this repository requires a working Caffe installation with `pycaffe` - more information on installing Caffe can be found [here](http://caffe.berkeleyvision.org/installation.html). Other Python dependencies can be installed with `pip install -r requirements.txt`.

#### Get VGG16 model

The VGG16 model is used for feature extraction. The model parameters can be downloaded into the `vgg` folder by running:

```bash
bash vgg/get_vgg.sh
```

The `vgg` folder contains a version of the VGG16 model prototxt file that is modified to remove all fully-connected layers. This is necessary in order to apply the model fully convolutionally. It is modified from the original version hosted here: [http://www.robots.ox.ac.uk/~vgg/research/very_deep/](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).

#### Get Oxford dataset

The Oxford dataset can be downloaded to the `oxford` folder by running:

```bash
bash oxford/get_oxford.sh
```

#### Get Paris dataset

The Paris dataset can be downloaded to the `paris` folder by running:

```bash
bash paris/get_paris.sh
```

#### Build evaluation script

The official C++ program provided by the Oxford group, `compute_ap.cpp`, for computing the mean average precision (mAP) on the retrieval benchmark is provided in this repository for convenience. It is modified to add an explicit include so that it can be compiled everywhere.

```bash
g++ -O compute_ap.cpp -o compute_ap
```

### Usage

Scripts are included to extract features for a directory of images, crop and extract features for query images in the retrieval datasets, and to evaluate aggregation schemes on the retrieval datasets. The scripts are designed to save raw features to disk after extraction. The evaluation script reads these raw features, applies an aggregation function, computes whitening/PCA, and runs the benchmark evalution. The aggregated features and the whitening/PCA params are not saved by default.

#### Extracting features

The following extracts features for both retrieval datasets. The `extract_queries.py` dataset is designed to read and crop the query images in the retrieval datasets. The `extract_features.py` script is general and can be used to extract features for a directory of images. Please refer to the command line arguments in the script for usage details. The following commands may take a while to run for all images in the dataset.

```bash
python extract_features.py --images oxford/data/* --out oxford/pool5
python extract_queries.py --dataset oxford
python extract_features.py --images paris/data/* --out paris/pool5
python extract_queries.py --dataset paris
```

#### Evaluation

Running `python evaluate.py` will evaluation mAP with the default settings defined in the script. Please refer to the command line arguments for options. Also of note is the `run_eval` function contained in the script. It can easily be used to run many evaluations or provide custom aggregation functions for experimentation.

### License

Code licensed under the Apache License, Version 2.0 license. See LICENSE file for terms.
