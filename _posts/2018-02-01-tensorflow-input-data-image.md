---
layout: post
title:  "TensorFlow: building an image data pipeline"
description: "Tutorial for the TensorFlow part of the starter code on how to input image data"
excerpt: "How to feed image data with tf.data"
author: "Olivier Moindrot"
date:   2018-01-24
mathjax: true
published: true
tags: tensorflow image tf.data
github: https://github.com/cs230-stanford/cs230-starter-code
module: Tutorials
---

<!-- TODO: comment -->
Building the input pipeline in a machine learning project is always long and painful, and can take more time than building the actual model.
In this tutorial we will learn how to use TensorFlow's Dataset module `tf.data` to build efficient pipelines for images.


<!-- TODO: keep the links? update them -->
This tutorial is among a series explaining the starter code:
#TODO: add here links to different posts
- [getting started][post-1]: installation, get started with the code for the projects
- this post: (TensorFlow) explain the global structure of the code
- [third post][tf-vision]: (Tensorflow - Vision) details for the computer vision example
- [fourth post][tf-nlp]: (Tensorflow - NLP) details for the NER example

#### Goals of this tutorial
- learn how to use `tf.data` and the best practices
- build an efficient pipeline for loading images and preprocessing them
<!-- TODO: a third one? -->


## An overview of tf.data

#TODO:
- explain `tf.data`
  - refer to tensorflow tutorials (we won't go into all the details)
  - high level view: with tf.estimator...
- explain `tf.data.Dataset`
- explain `tf.data.Iterator`
- explain initializable iterator, why we did this (training / eval)


## Building the image data pipeline

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


### Switch between train and validation


## Best practices
- explain shuffling: buffer size big enough
  - cf. stackoverflow post




### References

TODO:
- tf.data tutorials
- slides from @mrry
- our github tensorflow/image


<!-- Links -->
[github]: https://github.com/cs230-stanford/cs230-starter-code
[post-1]: https://cs230-stanford.github.io/project-starter-code.html
<!-- TODO: put correct link -->
[tf-post]: https://cs230-stanford.github.io/
<!-- TODO: put correct link -->
[tf-vision]: https://cs230-stanford.github.io/
<!-- TODO: put correct link -->
[tf-nlp]: https://cs230-stanford.github.io/
