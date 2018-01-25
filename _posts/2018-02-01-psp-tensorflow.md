---
layout: post
title:  "Starter Code: TensorFlow"
description: "Tutorial for the TensorFlow part of the starter code"
excerpt: "Tutorial for the TensorFlow part of the starter code"
author: "Guillaume Genthial, Olivier Moindrot"
date:   2018-01-24
mathjax: true
published: true
tags: tensorflow
github: https://github.com/cs230-stanford/cs230-starter-code/tensorflow
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
#TODO: add here links to different posts
- [getting started][post-1]: installation, get started with the code for the projects
- this post: (TensorFlow) explain the global structure of the code
- [third post][tf-vision]: (Tensorflow - Vision) details for the computer vision example
- [fourth post][tf-nlp]: (Tensorflow - NLP) details for the NER example

#### Goals of this tutorial
- learn more about TensorFlow
- learn how to correctly structure a deep learning project in TensorFlow
- fully understand the code to be able to use it for your projects


## Creating the input data pipeline

#TODO:
- explain `tf.data`
  - refer to tensorflow tutorials (we won't go into all the details)
- explain shuffling
- explain initializable iterator, why we did this (training / eval)


## Defining the model

#TODO:
- explain `tf.layers`
- explain train op + optimization
- explain metrics
- explain tensorboard
- explain `model_spec`
- explain `is_training` --> `model_fn` will be called twice


## Training and evaluation

### Training

#TODO
- put input_data + model together
- train + evaluate every epoch
- careful with the initialization in TensorFlow
  - initialize metric ops each epoch
  - initialize variables only once
- `sess.run()` without `feed_dict` (no placeholders)

### Evaluation

#TODO
- explain the `tf.metrics`
  - run on the whole dataset, `update_op`
- ...

## Other small stuff

#TODO
- logging
- saving
- Params

## Hyperparameter search

#TODO:
- explain how to use `hyperparam_search.py`
- tips for how to train?
  - put in a separate post??
  - we begin to have a lot of posts






<!-- Links -->
[github]: https://github.com/cs230-stanford/cs230-starter-code
[post-1]: https://cs230-stanford.github.io/project-starter-code.html
<!-- TODO: put correct link -->
[tf-post]: https://cs230-stanford.github.io/
<!-- TODO: put correct link -->
[tf-vision]: https://cs230-stanford.github.io/
<!-- TODO: put correct link -->
[tf-nlp]: https://cs230-stanford.github.io/
