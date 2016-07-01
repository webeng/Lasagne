# -*- coding: utf-8 -*-
import numpy as np

import theano
import theano.tensor as T

from lasagne.layers import MergeLayer


class CandidateOutputLayer(MergeLayer):
    """
    Layer of shape (batch_size, n_outputs)
    Parameters
    ----------
    output_layer: layer of shape (batch_size, n_outputs)
    candidate_layer: layer of shape (batch_size, max_n_candidates)
    candidate_mask_layer: layer of shape (batch_size, max_n_candidates)
    """
    def __init__(self, output_layer, candidate_layer, candidate_mask_layer,
                 non_linearity=T.nnet.softmax, name=None):
        MergeLayer.__init__(self, [output_layer, candidate_layer,
                                   candidate_mask_layer], name=name)
        self.non_linearity = non_linearity

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):

        # out: (batch_size, n_entities)
        # cand: (batch_size, n_candidates)
        # cand_mask: (batch_size, n_candidates)
        out, cand, cand_mask = inputs

        n_entities = self.input_shapes[0][1]
        is_candidate = T.eq(
            T.arange(n_entities, dtype='int32')[None, None, :],
            T.switch(cand_mask, cand,
                     -T.ones_like(cand))[:, :, None]).sum(axis=1)

        out = T.switch(is_candidate, out, -1000 * T.ones_like(out))

        return self.non_linearity(out)
