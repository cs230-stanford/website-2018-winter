---
layout: post
title:  "Introducing the Project Starter Code"
description: "Tutorial for the CS230 project starter code: introduction and installation"
excerpt: "Introduction and installation"
author: "Guillaume Genthial, Olivier Moindrot, Surag Nair"
date:   2018-01-24
mathjax: true
published: true
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

- this post: installation, get started with the code for the projects
- [second post][tf-post]: (TensorFlow) explain the global structure of the code
- [third post][tf-vision]: (Tensorflow - Vision) details for the computer vision example
- [fourth post][tf-nlp]: (Tensorflow - NLP) details for the NER example

#### Goals of the starter code

- help students kickstart their project with a working codebase
- in each tensorflow and pytorch, give two examples of a structured project: one for a vision task, one for a NLP task
- through this codebase, explain and demonstrate the best practices for structuring a deep learning project



---

## Installation

Each of the four examples (TensorFlow / PyTorch + Vision / NLP) is self-contained and can be used independently of the others.

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

Run `deactivate` if you want to leave the virtual environment. Next time you want to work on the project, just re-run `source .env/bin/activate`.

#### If you have a GPU

TODO: add instructions if GPU
- for tensorflow, `pip install tensorflow-gpu`
- for pytorch, ??


### Download the data

#### Vision

*All instructions can be found in the [`tensorflow/vision/README.md`](https://github.com/cs230-stanford/cs230-starter-code/blob/master/tensorflow/vision/README.md)*

For the vision example, we will used the SIGNS dataset created for this class. The dataset is hosted on google drive, download it [here][SIGNS].

This will download the SIGNS dataset (~1.1 GB) containing photos of hands signs making numbers between 0 and 5.
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

The images are named following `{label}_IMG_{id}.jpg` where the label is in `[0, 5]`.
The training set contains 1,080 images and the test set contains 120 images.

Once the download is complete, move the dataset into `data/SIGNS`. Run the script `build_dataset.py` which will resize the images to size `(64, 64)`. The new reiszed dataset will be located by default in `data/64x64_SIGNS`.


#### Natural Language Processing (NLP)

*All instructions can be found in the [`tensorflow/nlp/README.md`](https://github.com/cs230-stanford/cs230-starter-code/blob/master/tensorflow/nlp/README.md)*

We provide a small subset of the kaggle dataset (30 sentences) for testing in `data/small` but you are encouraged to download the original version on the [Kaggle](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data) website.

1. __[kaggle] Download the dataset__ `ner_dataset.csv` on [Kaggle](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data) and save it under the `nlp/data/kaggle` directory. Make sure you download the simple version `ner_dataset.csv` and NOT the full version `ner.csv`.

2. __[kaggle] Build the dataset__ Run the following script
```
python build_kaggle_dataset.py
```
It will extract the sentences and labels from the dataset, split it into train / test / dev and save it in a convenient format for our model. Here is the structure of the data
```
kaggle/
    train/
        sentences.txt
        labels.txt
    test/
        sentences.txt
        labels.txt
    dev/
        sentences.txt
        labels.txt
```
*Debug* If you get some errors, check that you downloaded the right file and saved it in the right directory. If you have issues with encoding, try running the script with python 2.7.

3. __[kaggle and small] Build the vocabulary__ For both datasets, `data/small` and `data/kaggle` you need to build the vocabulary, with
```
python build_vocab.py --data_dir="data/small"
```
or
```
python build_vocab.py --data_dir="data/kaggle"
```

---

## Structure of the code

#TODO: copy this to tensorflow / pytorch post and add more details (model folder)
The code for each example shares a common structure:
```
data/
    train/
    dev/
    test/
experiments/
model/
    *.py
build_dataset.py
train.py
search_hyperparams.py
synthesize_results.py
evaluate.py
```

<!-- TODO: check that the structure is still this -->
Here is each file or directory's purpose:
- `data/`: will contain all the data of the project (generally not stored on github), with an explicit train/dev/test split
- `experiments`: contains the different experiments (will be explained in the following section)
- `model/`: module defining the model and functions used in train or eval. Different for our PyTorch and TensorFlow examples
  <!--- `model/input_fn.py`: where you define the input data pipeline-->
  <!--- `model/model_fn.py`: creates the deep learning model-->
  <!--- `model/utils.py`: utility functions for handling hyperparams / logging-->
  <!--- `model/training.py`: utility functions to train a model-->
  <!--- `model/evaluation.py`: utility functions to evaluate a model-->
- `build_dataset.py`: creates or transforms the dataset, build the split into train/dev/test
- `train.py`: train the model on the input data, and evaluate each epoch on the dev set
- `search_hyperparams.py`: run `train.py` multiple times with different hyperparameters
- `synthesize_results.py`: explore different experiments in a directory and display a nice table of the results
- `evaluate.py`: evaluate the model on the test set (should be run once at the end of your project)

__Files that you'll need to modify (at first) are into the `model` module__.
You should only modify the model definition and the data pipeline at first.

---

## Running experiments

<!-- TODO: add ### titles for clearer layout? -->

Now that you have understood the structure of the code, we can try to train a model on the data, using the `train.py` script:
```bash
python train.py --model_dir experiments/base_model
```

We need to pass the model directory in argument, where the hyperparameters are stored in a json file named `params.json`.
Different experiments will be stored in different directories, each with their own `params.json` file. Here is an example:

`experiments/base_model/params.json`:
```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 20
}
```

The structure of `experiments` after running a few different models might look like this (try to give meaningful names to the directories):
```
experiments/
    base_model/
        params.json
        ...
    learning_rate/
        lr_0.1/
            params.json
        lr_0.01/
            params.json
    batch_norm/
        params.json
```

Each directory after training will contain multiple things:
- `params.json`: the list of hyperparameters, in json format
- `train.log`: the training log (everything we print to the console)
- `train_summaries`: train summaries for TensorBoard (TensorFlow only)
- `eval_summaries`: eval summaries for TensorBoard (TensorFlow only)
- `last_weights`: weights saved from the 5 last epochs
- `best_weights`: best weights (based on dev accuracy)


### Training and evaluation

We can now train an example model with the parameters provided in the configuration file `experiments/base_model/params.json`:
```bash
python train.py --model_dir experiments/base_model
```

Once training is done, we can evaluate on the test set:
```bash
python evaluate.py --model_dir experiments/base_model
```

This was just a quick example, so please refer to the detailed [TensorFlow][tf-post] / [PyTorch][pytorch-post] tutorials for an in-depth explanation of the code.


### Hyperparameters search

We provide an example that will call `train.py` with different values of learning rate. We first create a directory
```
experiments/
    learning_rate/
        params.jon
```

with a `params.json` file that contains the other hyperparameters. Then, by calling


```
python search_hyperparams.py --parent_dir="experiments/learning_rate"
```

It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`, like

```
experiments/
    learning_rate/
        learning_rate_0.001/
            metrics_eval_best_weights.json
        learning_rate_0.01/
            metrics_eval_best_weights.json
        ...
```

### Display the results of multiple experiments

If you want to aggregate the metrics computed in each experiment (the `metrics_eval_best_weights.json` files), simply run

```
python synthesize_results.py --parent_dir="experiments/learning_rate"
```

It will display a table synthesizing the results like this that is compatible with markdown:

```
|                                               |   accuracy |      loss |
|:----------------------------------------------|-----------:|----------:|
| experiments/base_model                        |   0.989    | 0.0550    |
| experiments/learning_rate/learning_rate_0.01  |   0.939    | 0.0324    |
| experiments/learning_rate/learning_rate_0.001 |   0.979    | 0.0623    |
```



### Tensorflow or PyTorch ?

Both framework have their pros and cons:

__Tensorflow__
- mature, most of the models and layers are already implemented in the library.
- documented and plenty of code / tutorials online
- built for large-scale deployment and used by a lot of companies
- has some very useful tools like Tensorboard for visualization
- but some ramp-up time is needed to understand some of the concepts (session, graph, variable scope, etc.) -- *(reason why we have a starter code that takes care of these subtleties)*
- transparent use of the GPU

__PyTorch__
- younger, but also well documented and fast-growing community
- more pythonic and numpy-like approach, easier to get used to the dynamic-graph paradigm
- designed for faster prototyping and research
- easy to debug


Which one will you [choose][matrix] ?

<div style="text-align: center;">
    <div style="display: inline-block; margin-right: 50px; margin-left: 50px;">
        <a href="https://cs230-stanford.github.io/psp-pytorch.html" style="text-decoration: none">
            <h3 style="background-color: #3168FC; color: white; border: 2px solid rgba(0,0,0,0.4); border-radius: 25px; padding: 0.2em 0.6em;">
                PyTorch
            </h3>
        </a>
    </div>
    <div style="display: inline-block; margin-right: 50px; margin-left: 50px" >
        <a href="https://cs230-stanford.github.io/tensorflow-psp.html" style="text-decoration: none">
            <h3 style="background-color: #B70B14; color: white; border: 2px solid rgba(0,0,0,0.4); border-radius: 25px; padding: 0.2em 0.6em;">
                Tensorflow
            </h3>
        </a>
    </div>
</div>


[github]: https://github.com/cs230-stanford/cs230-starter-code
[tf-post]: https://cs230-stanford.github.io/tensorflow-psp.html
[pytorch-post]: https://cs230-stanford.github.io/pytorch-psp.html
[tf-vision]: https://cs230-stanford.github.io/tensorflow-input-data-image.html
[tf-nlp]: https://cs230-stanford.github.io/tensorflow-input-data-text.html

<!-- TODO: add a public link -->
[SIGNS]: https://drive.google.com/drive/u/1/folders/19xqDh1dlfIs3G18DcDI1OvBom0T8AX6H
[matrix]: https://youtu.be/zE7PKRjrid4?t=1m26s
