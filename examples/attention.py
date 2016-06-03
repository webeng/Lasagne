#!/usr/bin/env python
# from __future__ import print_function

# import sys
# import os
# import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

document, u = T.lmatrix(), T.lmatrix()
vocabulary_size = 6

input_var = T.lmatrix()
# l_in = lasagne.layers.InputLayer(
#         shape=(10, 10), input_var=input_var)

a = np.random.rand(6, 20)
b = np.random.rand(6, 20)

# print a
# print b

a = lasagne.layers.InputLayer(
        shape=a.shape, input_var=a)

b = lasagne.layers.InputLayer(
        shape=b.shape, input_var=b)

l_in = lasagne.layers.ConcatLayer([a, b])
print "l_in shape {}".format(lasagne.layers.get_output_shape(l_in))
network = lasagne.layers.AttentionLayer(l_in, vocabulary_size, nonlinearity=lasagne.nonlinearities.tanh)
output = lasagne.layers.get_output(l_in)
get_output_values = theano.function([], outputs=output)
#print get_output_values()
attention_output = network.get_output_for(get_output_values())
get_attention_output_values = theano.function([], outputs=attention_output)
print "get_attention_output_values {}".format(get_attention_output_values().shape)
print "get_output_shape {}".format(lasagne.layers.get_output_shape(network))
# output = lasagne.layers.get_output(network)
# output.reshape((2, 2, 3))
# lasagne.layers.ReshapeLayer(network, shape=(2, 2, 3))



# get_output_values = theano.function([], outputs=output)
# print get_output_values()

# al = lasagne.layers.AttentionLayer(l_in, document, u, vocabulary_size)
