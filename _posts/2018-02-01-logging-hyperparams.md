---
layout: post
title:  "Logging and Hyperparameters"
description: "Best practice to log, load hyperparameters and do random search"
excerpt: "Best practice to log, load hyperparameters and do random search"
author: "Guillaume Genthial, Olivier Moindrot"
date:   2018-01-24
mathjax: true
published: true
tags: best-practice
github: https://github.com/cs230-stanford/cs230-starter-code
module: Tutorials
---

TODO: description

This tutorial is among a series of tutorials explaining how to structure a deep learning project.
Please see the full list of posts on the [main page][main].


__Table of Content__

* TOC
{:toc}

## Logging

A common problem when building a project is to forget about logging. In other words, as long as you write stuff in files and print things to the shell, people assume they're going to be fine. A better practice is to write __everything__ that you print to the terminal in a `log` file.

That's why in `train.py` and `evaluate.py` we initialize a `logger` using the built-in `logging` package with:

```python
# Set the logger
set_logger(os.path.join(args.model_dir, 'train.log'))
```
The `set_logger` function is defined in `utils.py`.

For instance during training it will create a `train.log` file in `experiments/base_model/`.
You don't have to worry too much about how we set it.
Whenever you want to print somehting, use `logging.info` instead of `print`:

```python
logging.info("It will be printed both to the Terminal and written in the .log file")
```


That way, you'll be able to both see it in the Terminal and remember it in the future when you'll need to read the `train.log` file.


## Loading hyperparameters from a configuration file

You'll quickly realize when doing a final project or any research project that you'll need a way to specify some parameters to your model. You have different sorts of parameters (not all of them are necessary):
- hyperparameters for the model: number of layers, number of channels...
- hyperparameters for the training: number of epochs, learning rate...
- parameters for the dataset: size of the dataset, size of the vocabulary for text...
- useful parameters: how often to save the model, how often to log to plot the loss...


There are multiple ways to load the parameters:

1. Use the `argparse` module as we do to specify the `data_dir` for instance
```python
parser.add_argument('--data_dir', default='data/',
                       help="Directory containing the dataset")
```
This quickly becomes unmanageable. Plus, how do you even keep track of the parameters if you want to go back to a previous experiment ?

2. Hard-code the parameters in an other `params.py` file and via some import at the begining of your `train.py` file for instance, get these parameters. Again, you'll need to find a way to save your config, and this is not very clean.

3. Write all your parameters in a file (we used `.json` but could be anything else) and store this file in the directory containing your experiment.
If you need to go back to your experiment later, you can quickly review which parameters yielded the performance etc.

We chose to take this third approach in our code. We define a class `Params` in `utils.py`.

Loading the parameters is as simple as writing

```python
params = Params("experiments/base_model/params.json")
```


and if your `params.json` file looks like

```json
{
    "model_version": "baseline",

    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 10
}
```


you'll be able to access the different entries with

```python
params.model_version
```
> In your code, once your params object is initialized, you can update it with another `.json` file with the `params.update("other_params.json")` method.

Later, in your code, for example when you define your model, you can thus do something like

```python
if params.model_version == "baseline":
    logits = build_model_baseline(inputs, params)
elif params.model_version == "simple_convolutions":
    logits = bulid_model_simple_convolutions(inputs, params)
```

which will be quite handy to have different functions and behaviors depending on a set of parameters !


## Hyperparameter search

An important part of any Machine Learning project is hyperparameter tuning. In other words, you want to see how your model performs on the development set on different sets of hyperparameters. There are basically 2 ways to implement this:

1. Have a python loop over the different set of hyperparameters and at each iteration of the loop, run the `train_and_evaluate(model_spec, params, ...)` function, like
```python
for lr in [0.1, 0.01, 0.001]:
    params.learning_rate = lr
    train_and_evaluate(model_spec, params, ...)
```

2. Have a more general script that will create a subfolder for each set of hyperparameteres and launch a training job using the `python train.py` command. While there is not much difference in the simplest setting, some more advanced clusters have some job managers and instead of running multiple `python train.py`, they instead do something like `job-manager-submit train.py` which will run the jobs concurrently, making the hyperparameter tuning much faster !
```python
for lr in [0.1, 0.01, 0.001]:
    params.learning_rate = lr
    # Create new experiment directory and save the relevant params.json
    subfolder = create_subfolder("lr_{}".format(lr))
    export_params_to_json(params, subfolder)
    # Launch a training in this directory -- it will call `train.py`
    lauch_training_job(model_dir=subfolder, ...)
```

This is what the `search_hyperparams.py` file does. It is basically a python script that runs other python scripts. Once all the sub-jobs have ended, you'll have the results of each experiment in a `metrics_eval_best_weights.json` file for each experiment directory.

```
learning_rate/
  params.json
  learning_rate_0.1/
      params.json
      metrics_eval_best_weights.json
  learning_rate_0.01/
      params.json
      metrics_eval_best_weights.json
```


and by running `python synthesize_results.py --model_dir experiments/learning_rate` you'll be able to gather the different metrics achieved for the different sets of hyperparameters !

<!-- Links -->
[main]: https://cs230-stanford.github.io
[github]: https://github.com/cs230-stanford/cs230-starter-code
[post-1]: https://cs230-stanford.github.io/project-starter-code.html
[tf-post]: https://cs230-stanford.github.io/tensorflow-psp.html
[tf-data]: https://cs230-stanford.github.io/tensorflow-input-data.html
