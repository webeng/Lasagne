import numpy as np
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer


__all__ = [
    "AttentionLayer",
]


class AttentionLayer(Layer):
    """
    lasagne.layers.AttentionLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    A fully connected layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, AttentionLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = AttentionLayer(l_in, num_units=50)

    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(self, incoming, num_units, Wym=init.GlorotUniform(),
                Wum=init.GlorotUniform(), Wms=init.GlorotUniform(),
                Wrg=init.GlorotUniform(), Wug=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.tanh,
                 **kwargs):
        super(AttentionLayer, self).__init__(incoming, **kwargs)
        # self.nonlinearity = (nonlinearities.identity if nonlinearity is None
        #                      else nonlinearity)
        self.nonlinearity = nonlinearity

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))
        # print num_inputs
        print self.input_shape

        # self.Wym = self.add_param(W, (num_inputs, vocabulary_size), name="Wym")
        # self.Wum = self.add_param(W, (num_inputs, vocabulary_size), name="Wum")

        # self.Wym = self.add_param(Wym, (self.input_shape[1] / 2, num_units), name="Wym")
        # self.Wum = self.add_param(Wum, (self.input_shape[1] / 2, num_units), name="Wum")

        self.Wym = self.add_param(Wym, (num_units, self.input_shape[0]), name="Wym")
        self.Wum = self.add_param(Wum, (num_units, self.input_shape[0]), name="Wum")
        self.Wms = self.add_param(Wms, (self.input_shape[1] / 2, num_units), name="Wms")

        self.Wrg = self.add_param(Wrg, (num_units, self.input_shape[0]), name="Wrg")
        self.Wug = self.add_param(Wug, (num_units, self.input_shape[0]), name="Wug")

        self.b = None
        # if b is None:
        #     self.b = None
        # else:
        #     self.b = self.add_param(b, (num_units,), name="b",
        #                             regularizable=False)
    # def __init__(self, incoming, num_units, W=init.GlorotUniform(),
    #              b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
    #              **kwargs):
    #     super(AttentionLayer, self).__init__(incoming, **kwargs)
    #     self.nonlinearity = (nonlinearities.identity if nonlinearity is None
    #                          else nonlinearity)

    #     self.num_units = num_units

    #     num_inputs = int(np.prod(self.input_shape[1:]))

    #     self.W = self.add_param(W, (num_inputs, num_units), name="W")
    #     if b is None:
    #         self.b = None
    #     else:
    #         self.b = self.add_param(b, (num_units,), name="b",
    #                                 regularizable=False)

    def get_output_shape_for(self, input_shape):
        print 'get_output_shape_for'
        return (self.num_units, input_shape[1] / 2)

    def get_output_for(self, input, **kwargs):
        input_reshape = input.reshape((2, self.input_shape[0], self.input_shape[1] / 2))
        input_q = input_reshape[0]
        input_doc = input_reshape[1]
        activation = T.dot(self.Wym, input_q) + T.dot(self.Wum, input_doc)

        M = self.nonlinearity(activation)
        # return M
        S = T.exp(T.dot(self.Wms, M))
        r = T.dot(input_doc, S)

        return self.nonlinearity(T.dot(self.Wrg, r) + T.dot(self.Wug, input_q))
