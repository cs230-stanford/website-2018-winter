---
layout: post
title:  "Building the data pipeline"
description: "Tutorial for the TensorFlow part of the starter code on how to input image data"
excerpt: "How to feed data to the model with tf.data"
author: "Olivier Moindrot, Guillaume Genthial"
date:   2018-01-24
mathjax: true
published: true
tags: tensorflow image nlp tf.data
github: https://github.com/cs230-stanford/cs230-starter-code
module: Tutorials
---
If you haven't read the previous post,

<div align="right"><a href="https://cs230-stanford.github.io/tensorflow-getting-started.html"><h3>> Introduction to Tensorflow</h3></a></div>

<br/>


__Motivation__

Building the input pipeline in a machine learning project is always long and painful, and can take more time than building the actual model.
In this tutorial we will learn how to use TensorFlow's Dataset module `tf.data` to build efficient pipelines for images and text.


<!-- TODO: keep the links? update them -->
This tutorial is among a series explaining the starter code:
<!-- TODO: add here links to different posts -->
- [first post][post-1]: installation, get started with the code for the projects
- [second post][post-2]: (TensorFlow) explain the global structure of the code
- __this post: (TensorFlow) how to build the data pipeline__
- [fourth post][post-4]: (Tensorflow) how to build the model and train it

__Goals of this tutorial__
- learn how to use `tf.data` and the best practices
- build an efficient pipeline for loading images and preprocessing them
- build an efficient pipeline for text, including how to build a vocabulary

__Table of contents__

* TOC
{:toc}



---

## An overview of tf.data

The `Dataset` API alows you to build an asynchronous, highly optimized data pipeline to prevent your GPU from [data starvation](https://www.tensorflow.org/performance/performance_guide#input_pipeline_optimization).
It loads data from the disk (images or text), applies optimized transformations, creates batches and sends it to the GPU. Former data pipelines made the GPU wait for the CPU to load the data, leading to performance issues.


Before explaining how `tf.data` works with a simple example, we'll share some great official resources:
- [API docs][api-tf-data] for `tf.data`
- [API docs][api-tf-contrib-data] for `tf.contrib.data`: new features still in beta mode. Contains useful functions that will soon be added to the main `tf.data`
- [Datasets Quick Start][quick-start-tf-data]: gentle introduction to `tf.data`
- [Programmer's guide][programmer-guide-tf-data]: more advanced and detailed guide to the best practices when using Datasets in TensorFlow
- [Performance guide][performance-guide]: advanced guide to improve performance of the data pipeline
- [Official blog post][blog-post-tf-data] introducing Datasets and Estimators: if you are using our [starter code][post-1], you can ignore the part about Estimators
- [Slides from the creator of tf.data][slides] explaining the API, best practices (don't forget to read the speaker notes below the slides)
- [Origin github issue][github-issue-tf-data] for Datasets: a bit of history on the origin of `tf.data`
- [Stackoverflow][stackoverflow] tag for the Datasets API

### Introduction to tf.data with a Text Example


Let's go over a quick example. Let's say we have a `file.txt` file containing sentences

```
I use Tensorflow
You use PyTorch
Both are great
```

Let's read this file with the `tf.data` API:

```python
dataset = tf.data.TextLineDataset("file.txt")
```

Let's try to iterate over it

```python
for line in dataset:
    print(line)
```

We get an error
```
> TypeError: 'TextLineDataset' object is not iterable
```

> Wait... What just happened ? I thought it was supposed to read the data.

### Iterators and transformations

What's really happening is that `dataset` is a node of the Tensorflow `Graph` that contains instructions to read the file. We need to initialize the graph and evaluate this node in a Session if we want to read it. While this may sound awfully complicated, this is quite the oposite : now, even the dataset object is a part of the graph, so you don't need to worry about how to feed the data into your model !

We need to add a few things to make it work. First, let's create an `iterator` object over the dataset

```
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
```

Now, `next_element` is a graph's node that will contain the next element of iterator over the Dataset at each execution. Now, let's run it

```python
with tf.Session() as sess:
    for i in range(3):
        print(sess.run(next_element))

>'I use Tensorflow'
>'You use PyTorch'
>'Both are great'
```


Now that you understand the idea behind the `tf.data` API, let's quickly review some more advanced tricks. First, you can easily apply transformations to your dataset. For instance, splitting words by space is as easy as adding one line
```python
dataset = dataset.map(lambda string: tf.string_split([string]).values)
```

Shuffling the dataset is also straightforward

```python
dataset = dataset.shuffle(buffer_size=3)
```

It will load elements 3 by 3 and shuffle them at each iteration.

You can also create batches

```
dataset = dataset.batch(2)
```

and pre-fetch the data (in other words, it will always have one batch ready to be loaded).

```
dataset = dataset.prefetch(1)
```

Now, let's see what our iterator has become

```python
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    print(sess.run(next_element))

>[['Both' 'are' 'great']
  ['You' 'use' 'PyTorch']]
 ```

and as you can see, we now have a batch created from the shuffled Dataset !

__All the nodes in the Graph are assumed to be batched: every Tensor will have `shape = [None, ...]` where None corresponds to the (unspecified) batch dimension__

### Why we use initializable iterators

As you'll see in the `input_fn.py` files, we decided to use an initializable iterator.

```python
dataset = tf.data.TextLineDataset("file.txt")
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
init_op = iterator.initializer
```

Its behavior is similar to the one above, but thanks to the `init_op` we can chose to "restart" from the beginning. This will become quite handy when we want to perform multiple epochs !

```python
with tf.Session() as sess:
    # Initialize the iterator
    sess.run(init_op)
    print(sess.run(next_element))
    print(sess.run(next_element))
    # Move the iterator back to the beginning
    sess.run(init_op)
    print(sess.run(next_element))

> 'I use Tensorflow'
  'You use PyTorch'
  'I use Tensorflow' # Iterator moved back at the beginning
```

> As we use only one session over the different epochs, we need to be able to restart the iterator. Some other approaches (like `tf.Estimator`) alleviate the need of using `initializable` iterators by creating a new session at each epoch. But this comes at a cost: the weights and the graph must be re-loaded and re-initialized with each call to `estimator.train()` or `estimator.evaluate()`.


### Where do I find the data pipeline in the code examples ?

The `model/input_fn.py` defines a function `input_fn` that returns a dictionnary that looks like

```python
images, labels = iterator.get_next()
iterator_init_op = iterator.initializer

inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
```

This dictionay of inputs will be passed to the model function, which we will detail in the [next post][post-4].


## Building an image data pipeline

#TOOD
- code examples
- preprocess(...)
- train_preprocess()
- eval_preprocess()



```python
def _parse_function(filename, label):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
        - Resize the image to size (224, 224)
    """
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, [224, 224])
    return resized_image, label
```


```python
def train_preprocess(image, label):
    """Image preprocessing for training.

    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label
```


```python
def input_fn(is_training, filenames, params):
    """Input function for the SIGNS dataset.

    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    # Labels will be between 0 and 5 included (6 classes in total)
    labels = [int(filename.split('/')[-1][0]) for filename in filenames]

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(_parse_function, num_parallel_calls=params.num_parallel_calls)
            .map(train_preprocess, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(_parse_function)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    if mode == 'predict':
        return {'images': images}

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs
```



### Data augmentation


## Building a text data pipeline

### Creating the vocabulary
- Creating the vocabulary
- Creating the dataset
- See post on train-dev-test split

### TensorFlow's Datasets for text


## Best practices

### Switch between train and validation
- how to have one tensor `inputs` for train and validation
- use initializable iterators
- also possible (and better?) to use one shot iterators, and have one for training, one for validation
  - (in our code, we used initializable iterators because we train multiple epochs, one by one)

### Shuffle and repeat

When training on a dataset, we need to repeat it for multiple epochs and we need to shuffle it.

- explain shuffling: buffer size big enough
  - cf. [stackoverflow post][stackoverflow-buffer-size]
  - best to already shuffle the filenames at the beginning
  - ex: if you have 100 files of text data, shuffle them before feeding to `tf.data`
- choose ordering between shuffle and repeat

- good to have shuffle + repeat at the beginning of the pipeline
  - because if shuffle has a big buffer


### Parallelization: using multiple threads
- add `num_parallel_calls` to `dataset.map()`

### Prefetch data

- enable consumer / producer overlap
  - consumer = deep learning model
  - producer = threads preprocessing data
- the model should never wait for the next batch of data
  - we want the GPU to be 100% used
- add `dataset.prefetch(1)` at the end of the pipeline (after batching) to always have one batch ready
  - could use more than 1 if the duration of the preprocessing varies a lot (it would average out the process, instead of sometimes waiting for longer batches)

### Order of the operations

To summarize, one good order for the different transformations is:
1. shuffle
2. repeat
3. map with the actual work (preprocessing, augmentation...)
4. batch
5. prefetch

<!--
TODO:
- our github tensorflow/image tensorflow/text
- seq2seq official tutorial -->


<br/>
<br/>
<br/>
<br/>

Now that we can input data to our model, let's actually see how we define it

<div align="right"><a href="https://cs230-stanford.github.io/tensorflow-model.html"><h3>> Creating and Training a Model</h3></a></div>

<!-- Links -->
[github]: https://github.com/cs230-stanford/cs230-starter-code
[post-1]: https://cs230-stanford.github.io/project-starter-code.html
<!-- TODO: put correct link -->
[post-2]: https://cs230-stanford.github.io/tensorflow-getting-started.html
[post-4]: https://cs230-stanford.github.io/tensorflow-model.html

<!-- Resources for tf.data -->
[api-tf-data]: https://www.tensorflow.org/api_docs/python/tf/data
[api-tf-contrib-data]: https://www.tensorflow.org/api_docs/python/tf/contrib/data
[quick-start-tf-data]: https://www.tensorflow.org/get_started/datasets_quickstart
[programmer-guide-tf-data]: https://www.tensorflow.org/programmers_guide/datasets
[performance-guide]: https://www.tensorflow.org/performance/performance_guide#input_pipeline_optimization
[blog-post-tf-data]: https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
[slides]: https://docs.google.com/presentation/d/16kHNtQslt-yuJ3w8GIx-eEH6t_AvFeQOchqGRFpAD7U
[github-issue-tf-data]: https://github.com/tensorflow/tensorflow/issues/7951
[stackoverflow]: https://stackoverflow.com/questions/tagged/tensorflow-datasets
[stackoverflow-buffer-size]: https://stackoverflow.com/a/48096625/5098368

