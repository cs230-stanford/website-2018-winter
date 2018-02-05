---
layout: post
title:  "Named Entity Recognition Tagging"
description: "NER Tagging in PyTorch"
excerpt: "Defining a Recurrent Network and Loading Text Data"
author: "Teaching assistants Surag Nair, Guillaume Genthial, Olivier Moindrot"
date:   2018-01-31
mathjax: true
published: true
tags: pytorch nlp
github: https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/nlp
module: Tutorials
---

<!-- TODO: comment -->

This post follows the [main post][post-1] announcing the CS230 Project Code Examples and the [PyTorch Introduction][pt-start]. In this post, we go through an example from Natural Language Processing, in which we learn how to load text data and perform Named Entity Recognition (NER) tagging for each token.


This tutorial is among a series explaining the code examples:
<!-- #TODO: add here links to different posts -->
- [getting started][post-1]: installation, getting started with the code for the projects
- [PyTorch Introduction][pt-start]: global structure of the PyTorch code examples
- [Vision][pt-vision]: predicting labels from images of hand signs
- **this post**: Named Entity Recognition (NER) tagging for sentences

__Goals of this tutorial__
- learn how to use PyTorch to load sequential data
- specify a recurrent neural network
- understand the key aspects of the code well-enough to modify it to suit your needs

__Table of Contents__

* TOC
{:toc}

<div style="height:5px;font-size:1px;">&nbsp;</div>

---
<div style="height:15px;font-size:1px;">&nbsp;</div>

### Problem Setup

We explore the problem of [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) (NER) tagging of sentences. The task is to tag each token in a given sentence with an appropriate tag such as Person, Location, etc.

```
John   lives in New   York
B-PER  O     O  B-LOC I-LOC
```

Our dataset will thus need to load both the sentences and labels. We will store those in 2 different files, a `sentence.txt` file containing the sentences (one per line) and a `labels.txt` containing the labels. For example:

```
# sentences.txt
John lives in New York
Where is John ?
```

```
# labels.txt
B-PER O O B-LOC I-LOC
O O B-PER O
```

Here we assume that we ran the `build_vocab.py` script that creates a vocabulary file in our `/data` directory. Running the script gives us one file for the words and one file for the labels. They will contain one token per line. For instance

```
# words.txt
John
lives
in
...
```

and

```
# tags.txt
B-PER
B-LOC
...
```

### Loading the Text Data

In NLP applications, a sentence is represented by the sequence of indices of the words in the sentence. For example if our vocabulary is `{'is':1, 'John':2, 'Where':3, '.':4, '?':5}` then the sentence "Where is John ?" is represented as `[3,1,2,5]`. We read the `words.txt` file and populate our vocabulary:

```python
vocab = {}
with open(words_path) as f:
  for i, l in enumerate(f.read().splitlines()):
    vocab[l] = i
``` 

In a similar way, we load a mapping `tag_map` from our labels from `tags.txt` to indices. Doing so gives us indices for labels in the range `[0,1,...,NUM_TAGS-1]`.

In addition to words read from English sentences, `words.txt` contains two special tokens: an `UNK` token to represent any word that is not present in the vocabulary, and a `PAD` token that is used as a filler token at the end of a sentence when one batch has sentences of unequal lengths. 

We are now ready to load our data. We read the sentences in our dataset (either train, validation or test) and convert them to a sequence of indices by looking up the vocabulary:

```python
train_sentences = []        
train_labels = []

with open(train_sentences_file) as f:
  for sentence in f.read().splitlines():
    # replace each token by its index if it is in vocab
    # else use index of UNK
    s = [vocab[token] if token in self.vocab 
         else vocab['UNK']
         for token in sentence.split(' ')]
    train_sentences.append(s)
    
with open(train_labels_file) as f:
  for sentence in f.read().splitlines():
    # replace each label by its index
    l = [tag_map[label] for label in sentence.split(' ')]
    train_labels.append(l)  
```
We can load the validation and test data in a similar fashion. 

<a name="batch"></a>
### Preparing a Batch

This is where it gets fun. When we sample a batch of sentences, not all the sentences usually have the same length. Let's say we have a batch of sentences `batch_sentences` that is a Python list of lists, with its corresponding `batch_tags` which has a tag for each token in `batch_sentences`. We convert them into a batch of PyTorch Variables as follows:

```python
# compute length of longest sentence in batch
batch_max_len = max([len(s) for s in batch_sentences])

# prepare a numpy array with the data, initializing the data with 'PAD' 
# and all labels with -1; initializing labels to -1 differentiates tokens 
# with tags from 'PAD' tokens
batch_data = vocab['PAD']*np.ones((len(batch_sentences), batch_max_len))
batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

# copy the data to the numpy array
for j in range(len(batch_sentences)):
    cur_len = len(batch_sentences[j])
    batch_data[j][:cur_len] = batch_sentences[j]
    batch_labels[j][:cur_len] = batch_tags[j]

# since all data are indices, we convert them to torch LongTensors
batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

# convert Tensors to Variables
batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
```

A lot of things happened in the above code. We first calculated the length of the longest sentence in the batch. We then initialized NumPy arrays of dimension `(num_sentences, batch_max_len)` for the sentence and labels, and filled them in from the lists. Since the values are indices (and not floats), PyTorch's Embedding layer expects inputs to be of the `Long` type. We hence convert them to `LongTensor`.

After filling them in, we observe that the sentences that are shorter than the longest sentence in the batch have the special token `PAD` to fill in the remaining space. Moreover, the `PAD` tokens, introduced as a result of packaging the sentences in a matrix, are assigned a label of -1. Doing so differentiates them from other tokens that have label indices in the range `[0,1,...,NUM_TAGS-1]`. This will be crucial when we calculate the loss for our model's prediction, and we'll come to that in a bit.

In our code, we package the above code in a custom data\_iterator function. Hyperparameters are stored in a data structure called "params". We can then use the generator as follows:
```python
# train_data contains train_sentences and train_labels
# params contains batch_size
train_iterator = data_iterator(train_data, params, shuffle=True)    

for _ in range(num_training_steps):
  batch_sentences, batch_labels = next(train_iterator)
  
  # pass through model, perform backpropagation and updates
  output_batch = model(train_batch)
  ...
```

### Recurrent Network Model

Now that we have figured out how to load our sentences and tags, let's have a look at the Recurrent Neural Network model. As mentioned in the [previous][pt-start] post, we first define the components of our model, followed by its functional form. Let's have a look at the `__init__` function for our model that takes in `(batch_size, batch_max_len)` dimensional data:

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self, params):
    super(Net, self).__init__()

    # maps each token to an embedding_dim vector
    self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

    # the LSTM takens embedded sentence
    self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim, batch_first=True)

    # fc layer transforms the output to give the final output layer
    self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)
```

We use an LSTM for the recurrent network. Before running the LSTM, we first transform each word in our sentence to a vector of dimension `embedding_dim`. We then run the LSTM over this sentence. Finally, we have a fully connected layer that transforms the output of the LSTM for each token to a distribution over tags. This is implemented in the forward propagation function:

```python
  def forward(self, s):
    # apply the embedding layer that maps each token to its embedding
    s = self.embedding(s)   # dim: batch_size x batch_max_len x embedding_dim
                
    # run the LSTM along the sentences of length batch_max_len
    s, _ = self.lstm(s)     # dim: batch_size x batch_max_len x lstm_hidden_dim                
                
    # reshape the Variable so that each row contains one token
    s = s.view(-1, s.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim
                
    # apply the fully connected layer and obtain the output for each token
    s = self.fc(s)          # dim: batch_size*batch_max_len x num_tags
    
    return F.log_softmax(s, dim=1)   # dim: batch_size*batch_max_len x num_tags
```

The embedding layer augments an extra dimension to our input which then has shape `(batch_size, batch_max_len, embedding_dim)`. We run it through the LSTM which gives an output for each token of length `lstm_hidden_dim`. In the next step, we open up the 3D Variable and reshape it such that we get the hidden state for each token, i.e. the new dimension is `(batch_size*batch_max_len, lstm_hidden_dim)`. Here the `-1` is implicitly inferred to be equal to `batch_size*batch_max_len`. The reason behind this reshaping is that the fully connected layer assumes a 2D input, with one example along each row. 

After the reshaping, we apply the fully connected layer which gives a vector of `NUM_TAGS` for each token in each sentence. The output is a log\_softmax over the tags for each token. We use log\_softmax since it is numerically more stable than first taking the softmax and then the log.

All that is left is to compute the loss. But there's a catch- we can't use a `torch.nn.loss` function straight out of the box because that would add the loss from the `PAD` tokens as well. Here's where the power of PyTorch comes into play- we can write our own custom loss function!

### Writing a Custom Loss Function

In the [section](#batch) on preparing batches, we ensured that the labels for the `PAD` tokens were set to `-1`. We can leverage this to filter out the `PAD` tokens when we compute the loss. Let us see how:
```python
def loss_fn(outputs, labels):
  # reshape labels to give a flat vector of length batch_size*seq_len
  labels = labels.view(-1)  
  
  # mask out 'PAD' tokens
  mask = (labels >= 0).float()
  
  # the number of tokens is the sum of elements in mask
  num_tokens = int(torch.sum(mask).data[0])
  
  # pick the values corresponding to labels and multiply by mask
  outputs = outputs[range(outputs.shape[0]), labels]*mask
  
  # cross entropy loss for all non 'PAD' tokens
  return -torch.sum(outputs)/num_tokens
```
The input labels has dimension `(batch_size, batch_max_len)` while outputs has dimension `(batch_size*batch_max_len, NUM_TAGS)`. We compute a mask using the fact that all `PAD` tokens in `labels` have the value `-1`. We then compute the Negative Log Likelihood Loss (remember the output from the network is already softmax-ed and log-ed!) for all the non `PAD` tokens. We can now compute derivates by simply calling `.backward()` on the loss returned by this function.

Remember, you can set a breakpoint using `pdb.set_trace()` at any place in the forward function, loss function or virtually anywhere and examine the dimensions of the Variables, tinker around and diagnose what's going wrong. That's the beauty of PyTorch :).

### Resources
- [Generating Names](http://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#sphx-glr-intermediate-char-rnn-generation-tutorial-py): a tutorial on character-level RNN 
- [Sequence to Sequence models](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py): a tutorial on translation

<div style="height:5px;font-size:1px;">&nbsp;</div>

---

<div style="height:15px;font-size:1px;">&nbsp;</div>

That concludes the description of the PyTorch NLP code example. If you haven't, take a look at the [Vision][pt-vision] example to understand how we load data and define models for images
 
<!-- TODO : have convention for links to code as well as actually include some links -->
<!-- Links -->
[github]: https://github.com/cs230-stanford/cs230-code-examples
[post-1]: https://cs230-stanford.github.io/project-code-examples.html
[pt-start]: https://cs230-stanford.github.io/pytorch-getting-started.html
[pt-vision]: https://cs230-stanford.github.io/pytorch-vision.html
