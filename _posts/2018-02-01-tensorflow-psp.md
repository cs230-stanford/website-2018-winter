---
layout: post
title:  "TensorFlow: starter-code"
description: "Tutorial for the TensorFlow part of the starter code"
excerpt: "Introduction to Tensorflow"
author: "Guillaume Genthial, Olivier Moindrot"
date:   2018-01-24
mathjax: true
published: true
tags: tensorflow
github: https://github.com/cs230-stanford/cs230-starter-code/tree/master/tensorflow
module: Tutorials
---

<!-- TODO: comment -->

This post follows the [main post][post-1] announcing the CS230 Project Starter Code.
We will explain here the TensorFlow part of the code, in our [github repository][github].

```
tensorflow/
    vision/
    nlp/
```

This tutorial is among a series explaining the starter code:
<!-- #TODO: add here links to different posts -->
- [getting started][post-1]: installation, get started with the code for the projects
- this post: (TensorFlow) explain the global structure of the code
- [third post][tf-vision]: (Tensorflow - Vision) details for the computer vision example
- [fourth post][tf-nlp]: (Tensorflow - NLP) details for the NER example

#### Goals of this tutorial
- learn more about TensorFlow
- learn how to correctly structure a deep learning project in TensorFlow
- fully understand the code to be able to use it for your projects


## Resources

For an official __introduction__ to the tensorflow concepts of `Graph()` and `Session()`, check out the [official introduction on tensorflow.org](https://www.tensorflow.org/get_started/get_started#tensorflow_core_tutorial).

For a __simple example on MNIST__, read [the official tutorial](https://www.tensorflow.org/get_started/mnist/beginners), but keep in mind that some of the techniques are not recommended for big projects (they use `placeholders` instead of the new `tf.data` pipeline, don't use `tf.layers`, etc.).

For a more __detailed tour__ of Tensorflow, reading the [programmer's guide](https://www.tensorflow.org/programmers_guide/) is definitely worth the time. You'll learn more about Tensors, Variables, Graphs and Session, as well as the saving mechanism or how to import data.

For a __more advanced use__ with concrete examples and code, we recommend reading [the relevant tutorials](https://www.tensorflow.org/tutorials/) for your project. You'll find good code and explanations, going from [sequence-to-sequence in Tensorflow](https://www.tensorflow.org/tutorials/seq2seq) to an [introduction to TF layers for convolutionnal Neural Nets](https://www.tensorflow.org/tutorials/layers#getting_started).

You might also be interested in [Stanford's CS20 class: Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/) and its [github repo](https://github.com/chiphuyen/stanford-tensorflow-tutorials) containing some cool examples.


## Getting Started

The code for each Tensorflow example shares a common structure:
```
data/
experiments/
model/
    input_fn.py
    model_fn.py
    utils.py
    training.py
    evaluation.py
train.py
search_hyperparams.py
synthesize_results.py
evaluate.py
```

Here is each `model/` file purpose:
  - `model/input_fn.py`: where you define the input data pipeline
  - `model/model_fn.py`: creates the deep learning model
  - `model/utils.py`: utility functions for handling hyperparams / logging
  - `model/training.py`: utility functions to train a model
  - `model/evaluation.py`: utility functions to evaluate a model

We recommend reading through `train.py` to get a high-level overview.

Once you get the high-level idea, depending on your task and dataset, you might want to modify
- `model/model_fn.py` to change the model, i.e. how you transform your input into your prediction as well as your loss, etc.
- `model/input_fn` to change the way you feed data to the model.
- `train.py` and `evaluate.py` to change the story-line (maybe you need to change the filenames, load a vocabulary, etc.)

Once you get something working for your dataset, feel free to edit any part of the code to suit your own needs.


## Creating the input data pipeline

<!-- #TODO: -->
- explain `tf.data`
  - refer to tensorflow tutorials (we won't go into all the details)
- explain shuffling
- explain initializable iterator, why we did this (training / eval)


## Defining the model

<!-- #TODO: -->
- explain `tf.layers`
- explain train op + optimization
- explain metrics
- explain tensorboard
- explain `model_spec`
- explain `is_training` --> `model_fn` will be called twice


## Training and evaluation

### Training

<!-- #TODO -->
- put input_data + model together
- train + evaluate every epoch
- careful with the initialization in TensorFlow
  - initialize metric ops each epoch
  - initialize variables only once
- `sess.run()` without `feed_dict` (no placeholders)

### Evaluation

<!-- #TODO -->

If you want to compute new metrics for which you can find a [tensorflow implementation](https://www.tensorflow.org/api_docs/python/tf/metrics), you can define it in the `model_fn.py` (add it to the `metrics` dictionnary). It will automatically be updated during the training and will be displayed at the end of each epoch.

- explain the `tf.metrics`
  - run on the whole dataset, `update_op`
- ...

## Other small stuff

<!-- #TODO -->
- logging
- saving
- Params

## Hyperparameter search

<!-- #TODO: -->
- explain how to use `hyperparam_search.py`
- tips for how to train?
  - put in a separate post??
  - we begin to have a lot of posts



<!-- Links -->
[github]: https://github.com/cs230-stanford/cs230-starter-code
[post-1]: https://cs230-stanford.github.io/project-starter-code.html
<!-- TODO: put correct link -->
[tf-post]: https://cs230-stanford.github.io/tensorflow-psp.html
<!-- TODO: put correct link -->
[tf-vision]: https://cs230-stanford.github.io/tensorflow-input-data-image.html
<!-- TODO: put correct link -->
[tf-nlp]: https://cs230-stanford.github.io/tensorflow-input-data-text.html
