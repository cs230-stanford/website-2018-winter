---
layout: post
title:  "Introducing the Project Starter Code"
description: "How to set up AWS for deep learning projects"
excerpt: "Introduction and installation"
author: "Russell Kaplan"
date:   2018-01-24
mathjax: true
published: false
tags: tensorflow pytorch
github: https://github.com/cs230-stanford/cs230-starter-code
module: Tutorials
---

__Table of Content__

* TOC
{:toc}


---

### Get your AWS credits

If no one on your team has requested AWS credit yet, please follow the instructions at [this piazza post](https://piazza.com/class/jc3wjzy24dyj?cid=192) to get your credits.

### Create a Deep Learning EC2 instance

Follow Amazon's (getting started guide)[https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/] for creating a Deep Learning instance. Be sure to pick the Ubuntu version of the deep learning AMI at the third screen. For the instance type, we recommend using p2.xlarge. Follow the instructions to SSH into the instance.

**IMPORTANT**: Be sure to **turn off your instance** when you are not using it! If you leave it running, you will be billed continuously for the hours it is left on and you will run out of credits very quickly.

<!-- TODO: May need a section on how to set up an EBS volume -->

### Clone the project starter code

It's not required to use our project starter code, but it might be helpful. (Some of you might be using starter code based on an existing GitHub repo instead, for example.) For an introduction to the starter code, see (our tutorial)[https://cs230-stanford.github.io/project-starter-code.html]. To clone, run this command inside your SSH session with the amazon server:
```
git clone https://github.com/cs230-stanford/cs230-starter-code.git
```


### Start training

You're ready to start training! Follow the instructions in the (project starter code tutorial)[https://cs230-stanford.github.io/project-starter-code.html] to start training a model. Good luck with your projects!