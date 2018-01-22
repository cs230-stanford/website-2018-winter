---
layout: post
title:  "Introducing the CS230 Project Starter Code"
description: "Tutorial for the CS230 project starter code"
excerpt: ""
author: "Guillaume Genthial, Olivier Moindrot, Surag Nair"
date:   2018-01-01
mathjax: true
published: false
tags: tensorflow pytorch
github: https://github.com/cs230-stanford/cs230-starter-code
module: Tutorials
---

<!-- TODO: comment -->

We are happy to introduce the project starter code for CS230. All the code used in the tutorial can be found on the corresponding [github repository][github]. The code has been well commented and detailed, so we recommend reading it entirely at some point if you want to use it for your project.

The code contains examples for TensorFlow and PyTorch, in vision and NLP. The structure of the repository is the following:
```
README.md
pytorch/
    vision/
    nlp/
tensorflow/
    vision/
    nlp/
```

This tutorial has multiple parts:
#TODO: add here links to different posts
- this post: installation, get started with the code for the projects
- [second post][tf-post]: (TensorFlow) explain the global structure of the code
- [third post][tf-vision]: (Tensorflow - Vision) details for the computer vision example
- [fourth post][tf-nlp]: (Tensorflow - NLP) details for the NER example

#### Goals of the starter code

- help students kickstart their project with a working codebase
- in each tensorflow and pytorch, give two examples of a structured project: one for a vision task, one for a NLP task
- through this codebase, explain and demonstrate the best practice for structuring a deep learning project



---

## Installation

Each of the four examples (TensorFlow / PyTorch + Vision / NLP) is self-contained and can be used independtly of the others.

Suppose you want to work with TensorFlow on a project involving computer vision. You can first clone the whole github repository and only keep the `tensorflow/vision` folder:

```bash
git clone https://github.com/cs230-stanford/cs230-starter-code
cd cs230-starter-code/tensorflow/vision
```

### Create your virtual environment
It is a good practice to have multiple virtual environments to work on different projects. Here we will use `python3` and install the requirements in the file `requirements.txt`.

**Installing Python 3**: To use `python3`, make sure to install version 3.5 or 3.6 on your local machine.
If you are on Mac OS X, you can do this using [Homebrew](https://brew.sh) with `brew install python3`. You can find instructions for Ubuntu [here](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-ubuntu-16-04).

**Virtual environment**: If we don't have it already, install `virtualenv` using `sudo pip install virtualenv` (or `pip install --user virtualenv` if you don't have sudo).
Here we create a virtual environment named `.env`.
```bash
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

Run `deactivate` if you want to leave the virtual environment.


### Download the data (for vision only)

For the vision example, we have provided a script `download_data.sh` located in the folder `data`.
You can download the dataset by running `sh download_data.sh` in your terminal.

This will download the SIGNS dataset (~1Gb) containing photos of hands signs making numbers between 0 and 5.
Here is the structure of the data:
```
SIGNS/
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...
```

The images are names following `{label}_IMG_{id}.jpg` where the label is in `[0, 5]`.
The training set contains 1,080 images and the test set contains 120 images.


---

## Structure of the code

The code for each example shares a common structure:
```
data/
    download_data.sh
experiments/
model/
    model.py
    utils.py
input_data.py
train.py
evaluate.py
hyperparams_search.py
```

<!-- TODO: check that the structure is still this -->
Here is each file or directory purpose:
- `data/`: will contain all the data of the project (generally not stored on github)
  - `data/download_data.sh`: script to download data. Makes it easy to clone the repo, download the data and be ready to work
- `experiments`: contains the different experiments run, the model weights... (will be explained in the following section)
- `model/`: all the files related to creating the model (here only one) and utilities
  - `model/model.py`: creates the deep learning model
  - `model/utils.py`: utility functions for handling hyperparams / logging
- `input_data.py`: where you define the input data pipeline
- `train.py`: train the model on the input data, and evaluate each epoch on the dev set
- `evaluate.py`: evaluate the model on the test set
- `hyperparams_search.py`: run `train.py` multiple times with different hyperparameters


---

## Running experiments

<!-- TODO: add ### titles for clearer layout? -->

Now that you have understood the structure of the code, we can try to train a model on the data, using the `train.py` script:
```bash
python train.py --model_dir experiments/test
```

We need to pass the model directory in argument, where the hyperparameters are stored in a json file named `params.json`.
Different experiments will be stored in different directories, each with their own `params.json` file. Here is an example:
```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 20,
}
```

The structure of `experiments` after running a few different models might look like this (try to give meaningful names to the directories):
```
experiments/
    base_model/
    learning_rate/
        lr_0.1/
        lr_0.01/
    batch_norm/
```

Each directory after training will contain multiple things:
- `params.json`: the list of hyperparameters, in json format
- `train.log`: the training log (everything we print)
- `train_summaries`: train summaries for TensorBoard (TensorFlow only)
- `eval_summaries`: eval summaries for TensorBoard (TensorFlow only)
- `last_weights`: weights saved from the 5 last epochs
- `best_weights`: best weights (based on dev accuracy)


### Training and evaluation

We can now train an example model with the parameters provided in `experiments/test/params.json`:
```bash
python train.py --model_dir experiments/test
```

Once training is done, we can evaluate on the test set:
```bash
python evaluate.py --model_dir experiments/test
```

This was just a quick example, so please refer to the detailed TensorFlow / PyTorch tutorials for an in-depth explanation of the code.




[github]: https://github.com/cs230-stanford/cs230-starter-code
<!-- TODO: put correct link -->
[tf-post]: https://cs230-stanford.github.io/
<!-- TODO: put correct link -->
[tf-vision]: https://cs230-stanford.github.io/
<!-- TODO: put correct link -->
[tf-nlp]: https://cs230-stanford.github.io/
