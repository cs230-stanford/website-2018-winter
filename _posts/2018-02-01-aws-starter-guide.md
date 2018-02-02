---
layout: post
title:  "AWS setup"
description: "How to set up AWS for deep learning projects"
excerpt: "How to set up AWS for deep learning projects"
author: "Teaching assistant Russell Kaplan"
date:   2018-01-24
mathjax: true
published: true
tags: tensorflow pytorch
github: https://github.com/cs230-stanford/cs230-code-examples
module: Tutorials
---

__Table of Content__

* TOC
{:toc}


---

### Get your AWS credits

For this winter 2018 session, AWS is offering GPU credits for CS230 students. If no one on your team has requested AWS credit yet, please follow the instructions on the AWS piazza post to get your credits.

### Create a Deep Learning EC2 instance

Follow Amazon's [getting started guide][aws-tutorial] for creating a __Deep Learning instance__. Be sure to pick the Ubuntu version of the deep learning Amazon Machine Images (AMI) at the third screen. For the instance type, we recommend using p2.xlarge. Follow the instructions to SSH into the instance.

**IMPORTANT**: Be sure to **turn off your instance** when you are not using it! If you leave it running, you will be billed continuously for the hours it is left on and you will run out of credits very quickly.

<!-- TODO: May need a section on how to set up an EBS volume -->

### Clone the project code examples

It's not required to base your project on the Project Code Examples, but it might be helpful. (Some of you might be using existing code from another GitHub repo instead, for example.)
For an introduction to the code examples, see [our tutorial][post-1]. To clone, run this command inside your SSH session with the amazon server:
```
git clone https://github.com/cs230-stanford/cs230-code-examples.git
```


### Start training

You're ready to start training! Follow the instructions in the [project tutorial][post-1] to start training a model. Good luck with your projects!



<!-- Links -->
[post-1]: https://cs230-stanford.github.io/project-code-examples.html
[aws-tutorial]: https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/
