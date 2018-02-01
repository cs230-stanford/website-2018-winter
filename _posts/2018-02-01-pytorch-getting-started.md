---
layout: post
title:  "Introduction to PyTorch Code Examples"
description: "Tutorial for the PyTorch part of the starter code"
excerpt: "An overview of training, models, loss functions and optimizers"
author: "Surag Nair, Guillaume Genthial, Olivier Moindrot"
date:   2018-01-31
mathjax: true
published: true
tags: pytorch
github: https://github.com/cs230-stanford/cs230-starter-code/tree/master/pytorch
module: Tutorials
---

<!-- TODO: comment -->

This post follows the [main post][post-1] announcing the CS230 Project Starter Code.
Here we explain some details of the PyTorch part of the code from our [github repository][github].

```
pytorch/
    vision/
    nlp/
```

This tutorial is among a series explaining the starter code:
<!-- #TODO: add here links to different posts -->
- [getting started][post-1]: installation, getting started with the code for the projects
- **this post**: global structure of the PyTorch code
- [Vision][pt-vision]: predicting labels from images of hand signs
- [NLP][pt-nlp]: Named Entity Recognition (NER) tagging for sentences

__Goals of this tutorial__
- learn more about PyTorch
- learn an example of how to correctly structure a deep learning project in PyTorch
- understand the key aspects of the code well-enough to modify it to suit your needs

__Table of Contents__

* TOC
{:toc}

<div style="height:5px;font-size:1px;">&nbsp;</div>

---
<div style="height:15px;font-size:1px;">&nbsp;</div>
### Resources

- The main PyTorch [homepage](http://pytorch.org/).
- The [official tutorials](http://pytorch.org/tutorials/) cover a wide variety of use cases- attention based sequence to sequence models, Deep Q-Networks, neural transfer and much more!
- Justin Johnson's [repository](https://github.com/jcjohnson/pytorch-examples) that introduces fundamental PyTorch concepts through self-contained examples.
- Tons of resources in this [list](https://github.com/ritchieng/the-incredible-pytorch).

### Code Layout

The code for each PyTorch example (Vision and NLP) shares a common structure:
```
data/
experiments/
model/
    net.py
    data_loader.py
train.py
evaluate.py
search_hyperparams.py
synthesize_results.py
evaluate.py
utils.py
```

  - `model/net.py`: specifies the neural network architecture, the loss function and evaluation metrics
  - `model/data_loader.py`: specifies how the data should be fed to the network
  - `train.py`: contains the main training loop 
  - `evaluate.py`: contains the main loop for evaluating the model
  - `utils.py`: utility functions for handling hyperparams/logging/storing model

We recommend reading through `train.py` to get a high-level overview.

Once you get the high-level idea, depending on your task and dataset, you might want to modify
- `model/net.py` to change the model, i.e. how you transform your input into your prediction as well as your loss, etc.
- `model/data_loader.py` to change the way you feed data to the model.
- `train.py` and `evaluate.py` to make changes specific to your problem, if required

Once you get something working for your dataset, feel free to edit any part of the code to suit your own needs.

### Core Training Step

Let's begin with a look at what the heart of our training algorithm looks like. The five lines below pass a batch of inputs through the model, calculate the loss, perform backpropagation and update the parameters.

```python
output_batch = model(train_batch)           # compute model output
loss = loss_fn(output_batch, labels_batch)  # calculate loss

optimizer.zero_grad()  # clear previous gradients
loss.backward()        # compute gradients of all variables wrt loss

optimizer.step()       # perform updates using calculated gradients
````

Each of the variables `train_batch`, `labels_batch`, `output_batch` and `loss` contains a PyTorch Tensor. PyTorch Tensors are very similar in behaviour to NumPy arrays and even support many of the same functions. More importantly, each of the variables is a PyTorch Variable- they remember the history of their computation. This allows you to do all sorts of funky things with them while allowing you to call `.backward()` to easily compute the derivatives. For more details, have a look at [this](http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html).

All the other code that we write is built around this- the exact specification of the model, how to fetch a batch of data and labels, computation of the loss and the details of the optimizer. In this post, we'll cover how to write a simple model in PyTorch, compute the loss and define an optimizer. The subsequent posts each cover a case of fetching data- one for image data and another for text data.

### Models in PyTorch
A model can be defined in PyTorch by subclassing the `torch.nn.Module` class. The model is defined in two steps. We first specify the parameters of the model, and then outline how it they are applied to the inputs. Here's an example of a simple 2 layer borrowed from [here](https://github.com/jcjohnson/pytorch-examples#pytorch-custom-nn-modules):

```python
class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear2 = torch.nn.Linear(H, D_out)
  
  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must 
    return a Variable of output data. We can use Modules defined in the 
    constructor as well as arbitrary operators on Variables.
    """
    h_relu = F.relu(self.linear1(x))    # F is torch.nn.functional module
    y_pred = self.linear2(h_relu)
    return y_pred
```

The `__init__` function initialises the two linear layers of the model. PyTorch takes care of the proper initialization of the parameters you specify. In the `forward` function, we first apply the first linear layer, apply ReLU activation and then apply the second linear layer. The module assumes that the first dimension of `x` is the batch size. If the input to the network is simply a vector of dimension 100, and the batch size is 32, then the dimension of `x` would be 32x100. Let's see an example of how to define a model and compute a forward pass:

```python
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 32, 100, 50, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(torch.randn(N, D_in))  # dim: 32 x 100

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Forward pass: Compute predicted y by passing x to the model
y_pred = model(x)   # dim: 32 x 10
```
More complex models follow the same layout, and we'll see two of them in the subsequent posts.

### Loss Function
<a name="lossfunc"></a>

PyTorch comes with many standard loss functions available for you to use in the `torch.nn` [module](http://pytorch.org/docs/master/nn.html#loss-functions). Here's a simple example of how to calculate Cross Entropy Loss. Let's say our model solves a multi-class classification problem with `C` labels. Then for a batch of size `N`, `out` is a PyTorch Variable of dimension `NxC`. We also have a `target` Variable of size `N`, where each element is the class for that example, i.e. a label in `[0,...,C-1]`. You can define the loss function and compute the loss as follows:


```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(out, target)
```

PyTorch makes it very easy to extend this and write your own custom loss function. We can write our own Cross Entropy Loss function as below (note the NumPy-esque syntax):
```python
def myCrossEntropyLoss(outputs, labels):
  batch_size = outputs.size()[0]            # batch_size
  outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
  outputs = outputs[range(num_examples), labels] # pick the values corresponding to the labels
  return -torch.sum(outputs)/num_examples
```

This was a fairly simple example of writing our own loss function. In the section on [NLP][pt-nlp], we'll see an interesting use of custom loss functions.

### Optimizer

The `torch.optim` [package](http://pytorch.org/docs/master/optim.html) provides an easy to use interface for common optimization algorithms. Defining your optimizer is really as simple as:

```python
# pick an SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

# or pick ADAM
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
```

The syntax is self-explanatory. You pass in the parameters of the model that need to be updated every iteration. You can also specify more complex methods such as per-layer or even per-parameter learning rates.

Once gradients have been computed using `loss.backward()`, calling `optimizer.step()` updates the parameters as defined by the optimization algorithm.

### Training vs Evaluation

Before training the model, it is imperative to call `model.train()`. Likewise, you must call `model.eval()` before testing the model. This corrects for the differences in dropout, batch normalisation during training and testing.

### Computing Metrics
By this stage you should be able to understand most of the code in `train.py` and `evaluate.py` (except how we fetch the data, which we'll come to in the subsequent posts). Apart from keeping an eye on the loss, it is also helpful to monitor other metrics such as accuracy and precision/recall. To do this, you can define your own metric functions for a batch of model outputs in the `model/net.py` file. In order to make it easier, we convert the PyTorch Variables into NumPy arrays before passing them into the metric functions. For a multi-class classification problem as set up in the section on [Loss Function](#lossfunc), we can write a function to compute accuracy using NumPy as:

```python
def accuracy(out, labels):
  outputs = np.argmax(out, axis=1)
  return np.sum(outputs==labels)/float(labels.size)
```

You can add your own metrics in the `model/net.py` file. Once you are done, simply add them to the `metrics` dictionary:
```python
metrics = {
  'accuracy': accuracy,
  # add your own custom metrics 
}
```

### Saving and Loading Models

We define utility functions to save and load models in `utils.py`. To save your model, call:
```python
state = {'epoch': epoch + 1,
         'state_dict': model.state_dict(),
         'optim_dict' : optimizer.state_dict()}
utils.save_checkpoint(state,
                      is_best=is_best,      # True if this is the model with best metrics
                      checkpoint=model_dir) # path to folder
```

`utils.py` internally uses the `torch.save(state, filepath)` method to save the state dictionary that is defined above. You can add more items to the dictionary, such as metrics. The `model.state_dict()` stores the parameters of the model and `optimizer.state_dict()` stores the state of the optimizer.

To load the saved state from a checkpoint, you may use:
```python
utils.load_checkpoint(restore_path, model, optimizer)
```

The `optimizer` argument is optional and you may choose to restart with a new optimizer. `load_checkpoint` internally loads the saved checkpoint and restores the model weights and the state of the optimizer.

### Using the GPU

Interspersed through the code you will find lines such as:
```python
> model = net.Net(params).cuda() if params.cuda else net.Net(params)

> if params.cuda:
     batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
```

PyTorch makes the use of the GPU explicit and transparent using these commands. Calling `.cuda()` on a model/Tensor/Variable sends it to the GPU. In order to train a model on the GPU, all the relevant parameters and Variables must be sent to the GPU using `.cuda()`. 

### Painless Debugging 

<div style="height:5px;font-size:1px;">&nbsp;</div>

---

<div style="height:15px;font-size:1px;">&nbsp;</div>

That concludes the introduction to the PyTorch code examples. You can proceed to the [Vision][pt-vision] example and/or the [NLP][pt-nlp] example to understand how we load data and define models specific to each domain.

<!-- TODO : have convention for links to code as well as actually include some links -->
<!-- Links -->
[github]: https://github.com/cs230-stanford/cs230-code-examples
[post-1]: https://cs230-stanford.github.io/project-code-examples.html
[pt-vision]: https://cs230-stanford.github.io/pytorch-vision.html
[pt-nlp]: https://cs230-stanford.github.io/pytorch-nlp.html
