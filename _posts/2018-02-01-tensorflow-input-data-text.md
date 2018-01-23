---
layout: post
title:  "TensorFlow: building a data pipeline for text"
description: "Tutorial for the TensorFlow part of the starter code on how to input text data"
excerpt: "Tutorial for the TensorFlow part of the starter code on how to input text data"
author: "Guillaume Genthial"
date:   2018-02-01
mathjax: true
published: false
tags: tensorflow nlp tf.data
github: https://github.com/cs230-stanford/cs230-starter-code
module: Tutorials
---

<!-- TODO: comment -->
Building the input pipeline in a machine learning project is always long and painful, and can take more time than building the actual model.  
In this tutorial we will learn how to use TensorFlow's Dataset module `tf.data` to build efficient pipelines for text.


<!-- TODO: keep the links? update them -->
This tutorial is among a series explaining the starter code:
#TODO: add here links to different posts
- [getting started][post-1]: installation, get started with the code for the projects
- this post: (TensorFlow) explain the global structure of the code
- [third post][tf-vision]: (Tensorflow - Vision) details for the computer vision example
- [fourth post][tf-nlp]: (Tensorflow - NLP) details for the NER example

#### Goals of this tutorial
<!-- TODO: change -->
- learn how to use `tf.data` for text and the best practices
- learn how to build a vocabulary and process a text dataset
- build an efficient pipeline for loading text and preprocessing it


## An overview of tf.data

<!-- Refer to other tutorial on image? -->


## Building the image data pipeline

#TOOD
- code examples


### Data augmentation


### Switch between train and validation


## Best practices
- explain shuffling: buffer size big enough
  - cf. stackoverflow post




### References

TODO:
- tf.data tutorials
- slides from @mrry
- seq2seq official tutorial
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
