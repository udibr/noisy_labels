from __future__ import print_function
from keras.layers import Dense
from keras import backend as K

class Channel(Dense):
    """
    Implement simple noise adaptation layer.

    References
        Goldberger & Ben-Reuven, Training deep neural-networks using a noise
        adaptation layer, ICLR 2017
        https://openreview.net/forum?id=H12GRgcxg

    # Arguments
        output_dim: int > 0
              default is input_dim which is known at build time
        See Dense layer for more arguments. There is no bias and the arguments
        `bias`, `b_regularizer`, `b_constraint` are not used.
    """

    def __init__(self, units = None, **kwargs):
        kwargs['use_bias'] = False
        if 'activation' not in kwargs:
            kwargs['activation'] = 'softmax'
        super(Channel, self).__init__(units, **kwargs)

    def build(self, input_shape):
        if self.units is None:
            self.units = input_shape[-1]
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
        channel_matrix = self.activation(self.kernel)

        # multiply the channel matrix with the baseline output:
        # channel_matrix[0,0] is the probability that baseline output 0 will get
        #  to channeled_output 0
        # channel_matrix[0,1] is the probability that baseline output 0 will get
        #  to channeled_output 1 ...
        # ...
        # channel_matrix[1,0] is the probability that baseline output 1 will get
        #  to channeled_output 0 ...
        #
        # we want output[b,0] = x[b,0] * channel_matrix[0,0] + \
        #                              x[b,1] * channel_matrix[1,0] + ...
        # so we do a dot product of axis -1 in x with axis 0 in channel_matrix
        return K.dot(x, channel_matrix)

if __name__ == '__main__':
    # * train a on MNIST in which the training labels are scrambled by a fixed
    #  permutation 46% of the time
    # * The baseline 3 layer MLP model gives an accuracy of 74% on MNIST test
    #  set which was not scrambled
    # * The confusion matrix of the noisy training data is computed
    # * We then add a customized Keras layer ([Channel](./channel.py)) to model
    #  the noise. This layer is initialized with the log of the confusion matrix
    #  (`channel_weights`):
    # ```python
    # channeled_output = Channel(name='channel',weights=[channel_weights])(
    #   baseline_output)
    # ```
    # * We continue training on the new output (`channeled_output`)
    # * The baseline output (`baseline_output`) has now an accuracy of 98%.
    #
    # For more information see the description of the [simple noise adaptation
    #  layer in the paper](https://openreview.net/forum?id=H12GRgcxg)
    import numpy as np
    import random

    seed = 42
    np.random.seed(seed)  # for reproducibility
    random.seed(seed)
    verbose = True

    # in case you dont have a GPU
    import os

    os.environ[
        'THEANO_FLAGS'] = 'device=cpu,floatX=float32'  # Use CPU on Theano
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU usage on tensorflow

    # # Data
    nb_classes = 10  # number of categories we classify. MNIST is 10 digits
    # input image dimensions. In CNN we think we have a "color" image with 1
    #  channel of color.
    # in MLP with flatten the pixels to img_rows*img_cols
    img_color, img_rows, img_cols = 1, 28, 28
    img_size = img_color * img_rows * img_cols

    from keras.datasets import mnist

    # keras has a built in tool that download the MNIST data set for you to
    #  `~/.keras/datasets/`
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print('MNIST training data set label distribution', np.bincount(y_train))
    print('test distribution', np.bincount(y_test))

    X_train = X_train.reshape(X_train.shape[0], img_size)
    X_test = X_test.reshape(X_test.shape[0], img_size)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # ## noisy labels
    NOISE_LEVEL = 0.46  # what part of training labels are permuted
    perm = np.array(
        [7, 9, 0, 4, 2, 1, 3, 5, 6, 8])  # noise permutation (from Reed)

    noise = perm[y_train]

    # replace some of the training labels with permuted (noise) labels.
    # make sure each categories receive an equal amount of noise
    from sklearn.model_selection import StratifiedShuffleSplit

    _, noise_idx = next(iter(StratifiedShuffleSplit(n_splits=1,
                                                    test_size=NOISE_LEVEL,
                                                    random_state=seed).split(
        X_train, y_train)))
    y_train_noise = y_train.copy()
    y_train_noise[noise_idx] = noise[noise_idx]

    # actual noise level
    1. - np.mean(y_train_noise == y_train)

    # split training data to training and validation
    # break the training set to 10% validation which we will use for early
    #  stopping.
    train_idx, val_idx = next(iter(
        StratifiedShuffleSplit(n_splits=1, test_size=0.1,
                               random_state=seed).split(X_train,
                                                        y_train_noise)))
    X_train_train = X_train[train_idx]
    y_train_train = y_train_noise[train_idx]
    X_train_val = X_train[val_idx]
    y_train_val = y_train_noise[val_idx]

    # # Model

    # ## baseline model
    # We use the `Sequential` model from keras
    # https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
    # as a single layer which computes the last hidden layer which we then use to
    # compute the baseline and as an input to the channel matrix

    nhiddens = [500, 300]
    DROPOUT = 0.5
    opt = 'adam'
    batch_size = 256
    patience = 4  # Early stopping patience
    epochs = 40  # number of epochs to train on

    from keras.models import Sequential

    hidden_layers = Sequential(name='hidden')

    from keras.layers import Dense, Dropout, Activation

    for i, nhidden in enumerate(nhiddens):
        hidden_layers.add(Dense(nhidden,
                                input_shape=(img_size,) if i == 0 else []))
        hidden_layers.add(Activation('relu'))
        hidden_layers.add(Dropout(DROPOUT))

    from keras.layers import Input

    train_inputs = Input(shape=(img_size,))

    last_hidden = hidden_layers(train_inputs)
    baseline_output = Dense(nb_classes, activation='softmax', name='baseline')(
        last_hidden)

    from keras.models import Model

    model = Model(inputs=train_inputs, outputs=baseline_output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


    # baseline model performance evaluation before training
    def eval(model, y_test=y_test):
        return dict(zip(model.metrics_names,
                        model.evaluate(X_test, y_test, verbose=False)))

    print(eval(model))

    # ### baseline training
    from keras.callbacks import EarlyStopping

    train_res = model.fit(X_train_train,
                          y_train_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          validation_data=(X_train_val,
                                           y_train_val),
                          callbacks=
                          [EarlyStopping(patience=patience, mode='min',
                                         verbose=verbose)]
                          )

    # ### baseline performance

    print(eval(model))

    # build confusion matrix (prediction,noisy_label)
    ybaseline_predict = model.predict(X_train, batch_size=batch_size)

    ybaseline_predict = np.argmax(ybaseline_predict, axis=-1)
    ybaseline_predict.shape

    baseline_confusion = np.zeros((nb_classes, nb_classes))
    for n, p in zip(y_train_noise, ybaseline_predict):
        baseline_confusion[p, n] += 1.

    # ## Simple channel model

    # ignore baseline loss in training
    BETA = 0.

    channel_weights = baseline_confusion.copy()
    channel_weights /= channel_weights.sum(axis=1, keepdims=True)
    # perm_bias_weights[prediction,noisy_label] = log(P(noisy_label|prediction))
    channel_weights = np.log(channel_weights + 1e-8)

    channeled_output = Channel(name='channel', weights=[channel_weights])(
        baseline_output)

    simple_model = Model(input=train_inputs,
                         output=[channeled_output, baseline_output])
    simple_model.compile(loss='sparse_categorical_crossentropy',
                         loss_weights=[1. - BETA, BETA],
                         optimizer=opt,
                         metrics=['accuracy'])

    train_res = simple_model.fit(X_train_train,
                                 [y_train_train, y_train_train],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbose,
                                 validation_data=(X_train_val,
                                                  [y_train_val, y_train_val]),
                                 callbacks=
                                 [EarlyStopping(patience=patience, mode='min',
                                                verbose=verbose)]
                                 )

    print(eval(simple_model, y_test=[y_test, y_test]))

