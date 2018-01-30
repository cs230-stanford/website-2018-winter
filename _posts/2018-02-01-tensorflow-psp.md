---
layout: post
title:  "Starter-code"
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

__Goals of this tutorial__
- learn more about TensorFlow
- learn how to correctly structure a deep learning project in TensorFlow
- fully understand the code to be able to use it for your projects

__Table of Content__

* TOC
{:toc}

---

## Getting Started

### Resources

For an official __introduction__ to the tensorflow concepts of `Graph()` and `Session()`, check out the [official introduction on tensorflow.org](https://www.tensorflow.org/get_started/get_started#tensorflow_core_tutorial).

For a __simple example on MNIST__, read [the official tutorial](https://www.tensorflow.org/get_started/mnist/beginners), but keep in mind that some of the techniques are not recommended for big projects (they use `placeholders` instead of the new `tf.data` pipeline, they don't use `tf.layers`, etc.).

For a more __detailed tour__ of Tensorflow, reading the [programmer's guide](https://www.tensorflow.org/programmers_guide/) is definitely worth the time. You'll learn more about Tensors, Variables, Graphs and Sessions, as well as the saving mechanism or how to import data.

For a __more advanced use__ with concrete examples and code, we recommend reading [the relevant tutorials](https://www.tensorflow.org/tutorials/) for your project. You'll find good code and explanations, going from [sequence-to-sequence in Tensorflow](https://www.tensorflow.org/tutorials/seq2seq) to an [introduction to TF layers for convolutionnal Neural Nets](https://www.tensorflow.org/tutorials/layers#getting_started).

You might also be interested in [Stanford's CS20 class: Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/) and its [github repo](https://github.com/chiphuyen/stanford-tensorflow-tutorials) containing some cool examples.

### Structure of the code

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


### Graph, Session and nodes

When designing a Model in Tensorflow, there are [basically 2 steps](https://www.tensorflow.org/get_started/get_started#tensorflow_core_tutorial)
1. building the computational graph, the nodes and operations and how they are connected to each other
2. evaluating / running this graph on some data

As an example of __step 1__, if we define a TF constant (=a graph node), when we print it, we get a *Tensor* object (= a node) and not its value

```python
x = tf.constant(1., dtype=tf.float32, name="my-node-x")
print(x)
> Tensor("my-node-x:0", shape=(), dtype=float32)
```

Now, let's get to __step 2__, and evaluate this node. We'll need to create a `tf.Session` that will take care of actually evaluating the graph

```python
with tf.Session() as sess:
    print(sess.run(x))
> 1.0
```


In the starter code,

- __step 1__ `model/input_fn.py` and `model/model_fn`

- __step 2__ `model/training.py` and `model/evaluation.py`

### A word about [variable scopes](https://www.tensorflow.org/versions/r0.12/how_tos/variable_scope/#the_problem)

When creating a node, Tensorflow will have a name for it. You can add a prefix to the nodes names. This is done with the `variable_scope` mechanism

```python
with tf.variable_scope('model'):
    x1 = tf.get_variable('x', [], dtype=tf.float32) # get or create variable with name 'model/x:0'
    print(x1)
> <tf.Variable 'model/x:0' shape=() dtype=float32_ref>
```

> What happens if I instantiate `x` twice ?

```python
with tf.variable_scope('model'):
    x2 = tf.get_variable('x', [], dtype=tf.float32)
> ValueError: Variable model/x already exists, disallowed.
```

When trying to create a new variable named `model/x`, we run into an Exception as a variable with the same name already exists. Thanks to this naming mechanism, you can actually controll which value you give to the different nodes, and at different points of your code, decide to have 2 python objects correspond to the same node !

```python
with tf.variable_scope('model', reuse=True):
    x2 = tf.get_variable('x', [], dtype=tf.float32)
    print(x2)
> <tf.Variable 'model/x:0' shape=() dtype=float32_ref>
```

We can check that they indeed have the same value
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # Initialize the Variables
    sess.run(tf.assign(x1, tf.constant(2.)))    # Change the value of x1
    print("x1 = ", sess.run(x1), " x2 = ", sess.run(x2))

> x1 =  2.0  x2 =  2.0
```

> Starter-code design choice: theoretically, the graphs you define for training and inference can be different, but they still need to share their weights. To remedy this issue, there are two possibilities:
1. re-build the graph, create a new session and reload the weights from some file when we switch between training and inference
2. create all the nodes for training and inference in the graph and make sure that the python code does not create the nodes twice by using the `reuse=True` trick explained above.
We decided to go for this option. For those interested in the problem of making training and eval graphs coexist, you can read this [discussion](https://www.tensorflow.org/tutorials/seq2seq#building_training_eval_and_inference_graphs).

---

## Creating the input data pipeline

You can read the [official tutorial](https://www.tensorflow.org/programmers_guide/datasets). The `Dataset` API alows you to build an asynchronous, highly optimized data pipeline to prevent your GPU from [data starvation](https://www.tensorflow.org/performance/performance_guide#input_pipeline_optimization). It loads data from the disk (images or text), applies optimized transformations, creates batches and sends it to the GPU. Former data pipelines made the GPU wait for the CPU to load the data, leading to performance issues.

### Introduction to `tf.data` with a Text Example


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
```python
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

It will load 3 elements, shuffle them, iterate over them, load 3 new elements, etc.

You can also create batches

```
dataset = dataset.batch(2)
```

and pre-fetch the data on the GPU (in other words, it will always have one batch ready on the GPU).

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

__All the nodes in the Graph are assumed to be batched, in other words every Tensor will have `shape = [None, ...]` where None corresponds to the (variable) batch dimension__

### Why we use `initializable` iterators

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
  'I use Tensorflow'
```

> As we use only one session over the different epochs, we need to be able to restart the iterator. Some other approaches (like `tf.Estimator`) alleviate the need of using `initializable` iterators by recreating a session at each new epoch. But this comes at a cost: the weights and the graph must be re-loaded and re-initialized with each new Session !

<!-- #TODO: -->
<!-- - explain `tf.data`
  - refer to tensorflow tutorials (we won't go into all the details)
- explain shuffling
- explain initializable iterator, why we did this (training / eval) -->


### Where do I find the data pipeline in the starter-code ?

The `model/input_fn.py` defines a function `input_fn` that returns a dictionnary that looks like

```python
images, labels = iterator.get_next()
iterator_init_op = iterator.initializer

inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
```

<!-- TODO : add the relevant links -->
Look at the [Computer Vision][tf-vision] or [NLP][tf-nlp] posts for more details. At this point, just assume that this `input` dictionnary is correctly initialized in `train.py` and passed to the `model_fn` that defines the model and the different training operations !

---

## Defining the model

Great, now we have this `input` dictionnary containing the Tensor corresponding to the data, let's explain how we build the model.


### Introduction to `tf.layers`

This high-level tensorflow API lets you build and prototype models in a few lines. You can have a look at the [official tutorial for computer vision](https://www.tensorflow.org/tutorials/layers), or at the [list of available layers](https://www.tensorflow.org/api_docs/python/tf/layers). The idea is quite simple so we'll just give an example.


Let's get an input Tensor with a similar mechanism than the one explained in the previous part. Remember that __None__ corresponds to the batch dimension.

```python
# shape = [None, 64, 64, 3]
images = inputs["images"]
```

Now, let's apply a convolution, a relu activation and a max-pooling. This is as simple as

```python
out = images
out = tf.layers.conv2d(out, 16, 3, padding='same')
out = tf.nn.relu(out)
out = tf.layers.max_pooling2d(out, 2, 2)
```

Finally, use this final tensor to predict the labels of the image (6 classes). We first need to reshape the output of the max-pooling to a vector

```python
# First, reshape the output into [batch_size, flat_size]
out = tf.reshape(out, [-1, 32 * 32 * 16])
# Now, logits is [batch_size, 6]
logits = tf.layers.dense(out, 6)
```
> Note the use of `-1`: Tensorflow will compute the corresponding dimension so that the total size is preserved.

The logits will be *unnormalized* scores for each example.

> In the starter code, the transformation from `inputs` to `logits` is done in the `build_model` function.

### Training ops


At this point, we have defined the `logits` of the model. We need to define our predictions, our loss, etc. You can have a look at the `model_fn` in `model/model_fn.py`.


```python
# Get the labels from the input data pipeline
labels = inputs['labels']
labels = tf.cast(labels, tf.int64)

# Define the prediction as the argmax of the scores
predictions = tf.argmax(logits, 1)

# Define the loss
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
```
>The `1` in `tf.argmax` tells Tensorflow to take the argmax on the axis = 1 (remember that axis = 0 is the batch dimension)

Now, let's use Tensorflow built-in functions to create nodes and operators that will train our model at each iteration !

```python
# Create an optimizer that will take care of the Gradient Descent
optimizer = tf.train.AdamOptimizer(0.01)

# Create the training operation
train_op = optimizer.minimize(loss)
```
> All these nodes are created by `model_fn` that returns a dictionnary `model_spec` containing all the necessary nodes and operators of the graph. This dictionnary will later be used for actually running the training operations etc.


And that's all ! Our model is ready to be trained. Remember that all the objects we defined so far are nodes or operators that are part of the Tensorflow graph. To evaluate them, we actually need to execute them in a session. Simply run

```python
with tf.Session() as sess:
    for i in range(num_batches):
        _, loss_val = sess.run([train_op, loss])
```
> Notice how we don't need to feed data to the session as the `tf.data` nodes automatically iterate over the dataset !
At every iteration of the loop, it will move to the next batch (remember the `tf.data` part), compute the loss, and execute the `train_op` that will perform one update of the weights !


For more details, have a look at the `model/training.py` file that defines the `train_and_evaluate` function.


### Putting `input_fn` and `model_fn` together


To summarize the different steps, we just give a high-level overview of what needs to be done in `train.py`

```python
# 1. Create the iterators over the Training and Evaluation datasets
train_inputs = input_fn(True, train_filenames, train_labels, params)
eval_inputs = input_fn(False, eval_filenames, eval_labels, params)

# 2. Define the model
logging.info("Creating the model...")
train_model_spec = model_fn('train', train_inputs, params)
eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

# 3. Train the model (where a session will actually run the different ops)
logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
```

The `train_and_evaluate` function performs a given number of epochs (= full pass of the `train_inputs`). At the end of each epoch, it evaluates the performance on the development set (`dev` or `train-dev` in the course material).

> Remember the discussion about different graphs for Training and evaluation. Here, notice how the `eval_model_spec` is given the `reuse=True` argument. It will make sure that the nodes of the Evaluation graph that must share weights with the Training graph __do__ share their weights.


### Evalution and `tf.metrics`


So far, we explained how we input data to the graph, how we define the different nodes and training ops, but we don't know (yet) how to compute some metrics on our dataset. There are basically 2 possibilities

1. __[run evaluation outside the Tensorflow graph]__ Evaluate the prediction over the training dataset by running `sess.run(prediction)` and use them to evaluate your model (without Tensorflow, with pure python code). This option can also be used if you need to write a file with all the predicitons and use a script (distributed by a conference for instance) to evaluate the performance of your model.
2. __[use Tensorflow]__ As the above method can be quite complicated for simple metrics, Tensorflow hopefully has some built-in tools to run evaluation. Again, we are going to create nodes and operations in the Graph. The concept is simple : we will use the `tf.metrics` API to build those, the idea being that we need to update the metric on each batch. At the end of the epoch, we can just query the updated metric !


We'll cover method 2 as this is the one we implemented in the starter code (but you can definitely go with option 1 by modifying `model/evaluation.py`). As most of the nodes of the graph, we define these *metrics* nodes and ops in `model/model_fn.py`.

```python
# Define the different metrics
with tf.variable_scope("metrics"):
    metrics = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions,
        'loss': tf.metrics.mean(loss)
    }

# Group the update ops for the tf.metrics, so that we can run only one op to update them all
update_metrics_op = tf.group(*[op for _, op in metrics.values()])

# Get the op to reset the local variables used in tf.metrics, for when we restart an epoch
metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
metrics_init_op = tf.variables_initializer(metric_variables)
```
> Notice that we define the metrics, a *grouped* update op and an initializer.
> Notice also how we define the metrics in a special `variable_scope` so that we can query the variables by name when we create the initializer !

Now, to evaluate the metrics on a dataset, we'll just need to run them in a session as we loop over our dataset

```python
with tf.Session() as sess:
    # Run the initializer to reset the metrics to zero
    sess.run(metrics_init_op)

    # Update the metrics over the dataset
    for _ in range(num_steps):
        sess.run(update_metrics_op)

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
```

And that's all ! If you want to compute new metrics for which you can find a [tensorflow implementation](https://www.tensorflow.org/api_docs/python/tf/metrics), you can define it in the `model_fn.py` (add it to the `metrics` dictionnary). It will automatically be updated during the training and will be displayed at the end of each epoch.

---

## Tensorflow Tips and Tricks

### Be careful with initialization

So far, we mentionned 3 different *initializer* operators.

```python
# 1. For all the variables (the weights etc.)
tf.global_variables_initializer()

# 2. For the dataset, so that we can chose to move the iterator back at the beginning
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
iterator_init_op = iterator.initializer

# 3. For the metrics variables, so that we can reset them to 0 at the beginning of each epoch
metrics_init_op = tf.variables_initializer(metric_variables)
```

During `train_and_evaluate` we perform the following schedule, all in one session

1. Loop over the training set, updating the weights and computing the metrics
2. Loop over the evaluation set, computing the metrics
3. Go back to step 1.

We thus need to run
- `tf.global_variable_initializer()` at the beginning of the first step 1.
- `iterator_init_op` at the beginning of every loop (step 1 or step 2)
- `metrics_init_op` at the beginning of every loop, to reset the metrics to zero (we don't want to compute the metrics averaged over the different epochs or different datasets !)

You can indeed check that this is what we do in `model/evaluation.py` or `model/training.py` when we actually run the graph !

### Saving


### Tensorboard and summaries


## Logging, Params and `search_hyperparams`

<!-- #TODO move to other post as this is in common with other posts -->


<!-- Links -->
[github]: https://github.com/cs230-stanford/cs230-starter-code
[post-1]: https://cs230-stanford.github.io/project-starter-code.html
<!-- TODO: put correct link -->
[tf-post]: https://cs230-stanford.github.io/tensorflow-psp.html
<!-- TODO: put correct link -->
[tf-vision]: https://cs230-stanford.github.io/tensorflow-input-data-image.html
<!-- TODO: put correct link -->
[tf-nlp]: https://cs230-stanford.github.io/tensorflow-input-data-text.html
