from keras.layers import Dense
from keras import backend as K

class Channel(Dense):
    """
    Implement simple noise adaptation layer.

    References
        Goldberger & Ben-Reuven, Training deep neural-networks using a noise adaptation layer, ICLR 2017
        https://openreview.net/forum?id=H12GRgcxg

    # Arguments
        output_dim: int > 0
              default is input_dim which is known at build time
        See Dense layer for more arguments. There is no bias and the arguments
        `bias`, `b_regularizer`, `b_constraint` are not used.
    """

    def __init__(self, output_dim = None, **kwargs):
        kwargs['bias'] = False
        if 'activation' not in kwargs:
            kwargs['activation'] = 'softmax'
        super(Channel, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        if self.output_dim is None:
            self.output_dim = input_shape[-1]
        super(Channel, self).build(input_shape)

    def call(self, x, mask=None):
        """
        :param x: the output of a baseline classifier model passed as an input
        It has a shape of (batch_size, input_dim) and
        baseline_output.sum(axis=-1) == 1
        :param mask: ignored
        :return: the baseline output corrected by a channel matrix
        """
        # convert W to the channel probability (stochastic) matrix
        # channel_matrix.sum(axis=-1) == 1
        # channel_matrix.shape == (input_dim, input_dim)
        channel_matrix = self.activation(self.W)

        # multiply the channel matrix with the baseline output:
        # channel_matrix[0,0] is the probability that baseline output 0 will get to channeled_output 0
        # channel_matrix[0,1] is the probability that baseline output 0 will get to channeled_output 1 ...
        # ...
        # channel_matrix[1,0] is the probability that baseline output 1 will get to channeled_output 0 ...
        #
        # we want output[b,0] = x[b,0] * channel_matrix[0,0] + \
        #                              x[b,1] * channel_matrix[1,0] + ...
        # so we do a dot product of axis -1 in x with axis 0 in channel_matrix
        return K.dot(x, channel_matrix)

    def get_config(self):
        """remove Dense parameters that are fixed"""
        config = super(Channel, self).get_config()
        del config['b_regularizer']
        del config['b_constraint']
        del config['bias']
        return config
