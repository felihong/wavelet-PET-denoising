# wavelet-PET-denoising

This repository contains source code for enhancing `low-dose PET` (positron-emission-tomography) images 
using `3D-UNet` model on `wavelet` domain. Project initialized by ultra-low-dose PET
[Grand Challenge](https://ultra-low-dose-pet.grand-challenge.org/), and all clinical PET data are collected from the 
grand-challenge platform.

## Project introduction

This project handles data & model workflow from training dataset preparation to model training, prediction and final
evaluation. All main components are implemented as Python classes, including:
* `DataLoader`: Handles functions and logic of creating training datasets from raw input PET images for baseline
model, `dataloader/baseline_data_loader.py`.
* Wavelet DataLoader: Handles functions and logic of creating training datasets from raw input PET images for wavelet 
model, `dataloader/wavelet_data_laoder.py`.
* SR_UnetGAN_3D: Contains the designed UNet-3D architecture and training
procedures, `networks/FCN_3D.py`.
* Predictor: Procedures of loading noisy PET image into test dataset and applying baseline model to the test, 
and processing transformations to acquire enhanced PET output, `prediction/baseline_inference.py`.
* WaveletPredictor: Procedures of loading noisy PET image into test dataset and applying wavelet model to the test, 
and processing transformations to acquire enhanced PET output,`prediction/wavelet_inference.py`
* Evaluator: Measures quality enhancement with respect to global and local metrics, `common/evaluation.py`.

A high-level architecture design is shown as follows:
![architecture](docs/architecture.jpg)


## Get started
We use `Python=3.5.2` as main interpreter, based on which multiple deep-learning and data science packages are installed. 
Use the following commands to clone the project:
```
git clone https://github.com/felihong/wavelet-PET-denoising.git
cd wavelet-PET-denoising
```
Afterwards, install `virtualenv` to create a new virtual environment:
```
pip install virtualenv
virtualenv --python python3.5.2 venv
```
Activate the created environment and install all required packages:
```
source ~/virtualenv/venv/bin/activate
pip install -r requirements.txt
```

## Dataset creation

## Submit training

## Evaluation