---
layout: post
title:  "Classifying Images of Hand Signs"
description: "Defining a Convolutional Network and Loading Image Signs"
excerpt: "Defining a Convolutional Network and Loading Image Data"
author: "Surag Nair, Guillaume Genthial, Olivier Moindrot"
date:   2018-01-31
mathjax: true
published: true
tags: pytorch vision
github: https://github.com/cs230-stanford/cs230-starter-code/tree/master/pytorch/vision
module: Tutorials
---

<!-- TODO: comment -->

This post follows the [main post][post-1] announcing the CS230 Project Starter Code and the [PyTorch Introduction][pt-start] to the Starter Code. In this post, we go through an example from Computer Vision, in which we learn how to load images of hand signs and classify them.


This tutorial is among a series explaining the starter code:
<!-- #TODO: add here links to different posts -->
- [getting started][post-1]: installation, getting started with the code for the projects
- [PyTorch Introduction][pt-start]: global structure of the PyTorch code examples
- **this post**: predicting labels from images of hand signs
- [NLP][pt-nlp]: Named Entity Recognition (NER) tagging for sentences

__Goals of this tutorial__
- learn how to use PyTorch to load image data efficiently
- specify a convolutional networ
- understand the key aspects of the code well-enough to modify it to suit your needs

__Table of Contents__

* TOC
{:toc}

<div style="height:5px;font-size:1px;">&nbsp;</div>

---
<div style="height:15px;font-size:1px;">&nbsp;</div>

### Problem Setup

We use images from the SIGNS dataset. Each image from this dataset is a picture of a hand making a sign that represents a number between 1 and 6. It is 1080 training images and 120 test images. In our example, we use images scaled down to size `64x64`.


### Making a PyTorch Dataset

`torch.utils.data` provides some nifty functionality for loading data. We use `torch.utils.data.Dataset`, which is an abstract class representing a dataset. To make our own SIGNSDataset, we need to inherit `Dataset` and override the following methods:
- `__len__`: so that `len(dataset)` returns the size of the dataset
- `__getitem__`: to support indexing using `dataset[i]` to get the ith image

We then define our class as below:
```python
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class SIGNSDataset(Dataset):
  def __init__(self, data_dir, transform):      
    # store filenames
    self.filenames = os.listdir(data_dir)
    self.filenames = [os.path.join(data_dir, f) for f in self.filenames]
      
    # the first character of the filename contains the label
    self.labels = [int(filename.split('/')[-1][0]) for 
                   filename in self.filenames]
    self.transform = transform
      
  def __len__(self):
    # return size of dataset
    return len(self.filenames)
      
  def __getitem__(self, idx):
    # open image, apply transforms and return with label
    image = Image.open(self.filenames[idx])  # PIL image
    image = self.transform(image)
    return image, self.labels[idx]
```

Notice that when we return an image-label pair using `__getitem__` we apply a `tranform` on the image. These transformations are a part of the `torchvision.transforms` [package](http://pytorch.org/docs/master/torchvision/transforms.html), that allow us to manipulate images easily. Consider the following composition of multiple transforms:

```python
train_transformer = transforms.Compose([
  transforms.Resize(64),              # resize the image to 64x64 
  transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
  transforms.ToTensor()])             # transform it into a PyTorch Tensor
```

When we apply `self.transform(image)` in `__getitem__`, we pass it through the above transformations before using it as a training example. The final output is a PyTorch Tensor. To augment the dataset during training, we also use the `RandomHorizontalFlip` transform when loading the image. We specify a similar `eval_transformer` for evaluation without the random flip. To load a `Dataset` object for the different splits of our data, we simply use:

```python
train_dataset = SIGNSDataset(train_data_path, train_transformer)
val_dataset = SIGNSDataset(val_data_path, eval_transformer)
test_dataset = SIGNSDataset(test_data_path, eval_transformer)
```

### Loading Batches of Data

`torch.utils.data.DataLoader` provides an interator that takes in a `Dataset` object and performs batching, shuffling and loading of the data in parallel using multiprocessing workers. This is crucial when images are big in size and take time to load. In such a case, our GPU can be left idling while the CPU fetches the images, applies the transforms and then loads them onto the GPU. In contrast, the DataLoader class asynchronously fetches the data and keeps it ready to be sent to the GPU. Initialising the `DataLoader` is quite easy:

```python
train_dataloader = DataLoader(SIGNSDataset(train_data_path, train_transformer), 
                              batch_size=params.batch_size, shuffle=True,
                              num_workers=params.num_workers)
```

We can then iterate through batches of examples as follows:
```python
for train_batch, labels_batch in train_dataloader:
  # wrap Tensors in Variables
  train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
  
  # pass through model, perform backpropagation and updates
  output_batch = model(train_batch)
  ...
```

Applying transformations on the data load them as PyTorch Tensors. Since we need to add them to our computational graph, we convert them to PyTorch Variables before passing them into the model. The `for` loop ends after one pass over the data, i.e. after one epoch. We can use similar data loaders for validation and test data.

### Convolutional Network Model

Now that we have figured out how to load our images, let's have a look at the *pièce de résitance*- the CNN model. As mentioned in the [previous][pt-start] post, we first define the components of our model, followed by its functional form. Let's have a look at the `__init__` function for our model that takes in a `3x64x64` image:

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    # we define convolutional layers 
    self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
            
    # 2 fully connected layers to transform the output of the convolution layers to the final output
    self.fc1 = nn.Linear(8*8*128, 128)
    self.fcbn1 = nn.BatchNorm1d(128)
    self.fc2 = nn.Linear(128, 6)       
    self.dropout_rate = params.dropout_rate
```

The first parameter to the convolutional filter `nn.Conv2d` is the number of input channels, the second is the number of output channels, and the third is the size of the square filter (`3x3` in this case). Similarly, the batch normalisation layer takes as input the number of channels for 2D images and the number of features in the 1D case. The fully connected `Linear` layers take  the input and output dimensions.

In this example, we explicitly specify each of the values. In order to make the initialisation of the model more flexible, you can pass in parameters such as image size to the `__init__` function and use that to specify the sizes. You must be very careful when specifying parameter dimensions, since mismatches will lead to errors in the forward propagation. Let's now look at the forward propagation:

```python
  def forward(self, s):
    # we apply the convolution layers, followed by batch normalisation, 
    # maxpool and relu x 3
    s = self.bn1(self.conv1(s))        # batch_size x 32 x 64 x 64
    s = F.relu(F.max_pool2d(s, 2))     # batch_size x 32 x 32 x 32
    s = self.bn2(self.conv2(s))        # batch_size x 64 x 32 x 32
    s = F.relu(F.max_pool2d(s, 2))     # batch_size x 64 x 16 x 16
    s = self.bn3(self.conv3(s))        # batch_size x 128 x 16 x 16
    s = F.relu(F.max_pool2d(s, 2))     # batch_size x 128 x 8 x 8
            
    # flatten the output for each image
    s = s.view(-1, 8*8*self.num_channels*4)  # batch_size x 8*8*128
            
    # apply 2 fully connected layers with dropout
    s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
        p=self.dropout_rate, training=self.training)    # batch_size x 128
    s = self.fc2(s)                                     # batch_size x 6
            
    return F.log_softmax(s, dim=1)
```

We pass the image through 3 layers of `conv > bn > max_pool > relu`, followed by flattening the image and then applying 2 fully connected layers. The output is a log_softmax over the 6 labels for each example in the batch.

And that's it! We use an appropriate loss function (Negative Loss Likelihood, since the output is already softmax-ed and log-ed) and train the model as discussed in the [previous][pt-start] post. Remember, you can set a breakpoint using `pdb.set_trace()` at any place in the forward function, examine the dimensions of the Variables, tinker around and diagnose what's going wrong. That's the beauty of PyTorch :).

### Resources
- [Data Loading and Processing Tutorial](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html): an official tutorial from the PyTorch website
- [ImageNet](https://github.com/pytorch/examples/blob/master/imagenet/main.py): Code for training on ImageNet in PyTorch

<div style="height:5px;font-size:1px;">&nbsp;</div>

---

<div style="height:15px;font-size:1px;">&nbsp;</div>

That concludes the description of the PyTorch Vision code example. You can proceed to the [NLP][pt-nlp] example to understand how we load data and define models for text.
 
<!-- TODO : have convention for links to code as well as actually include some links -->
<!-- Links -->
[github]: https://github.com/cs230-stanford/cs230-code-examples
[post-1]: https://cs230-stanford.github.io/project-code-examples.html
[pt-start]: https://cs230-stanford.github.io/pytorch-getting-started.html
[pt-nlp]: https://cs230-stanford.github.io/pytorch-nlp.html
