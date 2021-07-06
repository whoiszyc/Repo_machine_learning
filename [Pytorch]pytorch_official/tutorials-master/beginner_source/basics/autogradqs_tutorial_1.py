"""
`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ || 
`Tensors <tensorqs_tutorial.html>`_ || 
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
**Autograd** ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

Automatic Differentiation with ``torch.autograd``
=======================================

When training neural networks, the most frequently used algorithm is
**back propagation**. In this algorithm, parameters (model weights) are
adjusted according to the **gradient** of the loss function with respect
to the given parameter.

To compute those gradients, PyTorch has a built-in differentiation engine
called ``torch.autograd``. It supports automatic computation of gradient for any
computational graph.

Consider the simplest one-layer neural network, with input ``x``,
parameters ``w`` and ``b``, and some loss function. It can be defined in
PyTorch in the following manner:
"""

import torch

x = torch.ones(1) * 5  # input tensor
y = torch.zeros(1)  # expected output
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
z = torch.matmul(x, w)+b

# trditional loss functions
# Essentially, the loss is a tensor in terms of w and b that has the backward function for autograd
# loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# Thus, here instead of evaluate loss, we try to back propogate z directly
# in this case, b.grad should be also 1, and w.grad should equal to x
z.backward()
print(w.grad)
print(b.grad)

######################################################################
# Tensors, Functions and Computational graph
# ------------------------------------------
#
# This code defines the following **computational graph**:
#
# .. figure:: /_static/img/basics/comp-graph.png
#    :alt:
#
# In this network, ``w`` and ``b`` are **parameters**, which we need to
# optimize. Thus, we need to be able to compute the gradients of loss
# function with respect to those variables. In order to do that, we set
# the ``requires_grad`` property of those tensors.

#######################################################################
# .. note:: You can set the value of ``requires_grad`` when creating a
#           tensor, or later by using ``x.requires_grad_(True)`` method.

#######################################################################
# A function that we apply to tensors to construct computational graph is
# in fact an object of class ``Function``. This object knows how to
# compute the function in the *forward* direction, and also how to compute
# its derivative during the *backward propagation* step. A reference to
# the backward propagation function is stored in ``grad_fn`` property of a
# tensor. You can find more information of ``Function`` `in the
# documentation <https://pytorch.org/docs/stable/autograd.html#function>`__.
#


######################################################################
# Computing Gradients
# -------------------
#
# To optimize weights of parameters in the neural network, we need to
# compute the derivatives of our loss function with respect to parameters,
# namely, we need :math:`\frac{\partial loss}{\partial w}` and
# :math:`\frac{\partial loss}{\partial b}` under some fixed values of
# ``x`` and ``y``. To compute those derivatives, we call
# ``loss.backward()``, and then retrieve the values from ``w.grad`` and
# ``b.grad``:
