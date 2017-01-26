from __future__ import print_function  # python 2 or 3
import fasteners
import numpy as np
import random
import cPickle as pickle
import argparse
# os.environ['THEANO_FLAGS'] = 'device=gpu0,floatX=float32,lib.cnmem=1'  # Use GPU
# os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'  # Use CPU
# import theano
from keras.datasets import mnist, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


from keras.layers import Input, RepeatVector, Permute, Reshape, BatchNormalization, Lambda, K
from keras.models import Model
from keras.engine.topology import merge
from keras.regularizers import l2
from tqdm import tqdm
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
import time
from utils import load_weights
from keras.optimizers import SGD

parser = argparse.ArgumentParser(description='train/test classifier when some '
    'of the training labels permuted by a fixed noise permutation. '
    'Comparing Jacob method to [Reed](http://arxiv.org/pdf/1412.6596v3.pdf).'
    'Results are added to <FN>.results.pkl file. '
    'You can run several runs in parallel. ',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument('--FN', type=str, default='data/channel',
                   help="prefix to all files generated. "
                        "The results of the run are added to <FN>.results.pkl")
parser.add_argument('--down_sample', type=float, default=1,
                   help='what percentage of training data to use')
parser.add_argument('--seed', type=int, default=42,
                   help='to make the experiments reproducable and different')
parser.add_argument('--perm', type=str, default='reed',
                   help="What permutation to apply to some of the labels:\n"
                        "reed - use permuatrion from Reed's paper\n"
                        "random - select one random permutation to use on all training (w/o stationary points)\n"
                        "cyclic - use cyclic permutation.\n"
                        "weak - build a very weak classifier and use its predicted probabilities as the noisy labels.\n"
                        "weak_hard - build a very weak classifier and use its predicted labeling as the noisy labels.\n"
                        "strong - build a strong classifier and use its labeling as the noisy labels.\n"
                        "noise - use different random permuation on every noised label"
                    )
parser.add_argument('--cnn', action='store_true', default=False,
                   help='Use CNN for baseline model (default MLP)')
parser.add_argument('--beta', type=float, default=1,
                   help='The weight of the baseline loss '
                        '(If 1, compute only baseline and save weights). '
                        'Reed soft in his paper was with 0.95 and 0.8 for hard')
parser.add_argument('--model', type=str, default='complex',
                    choices=['simple','complex','reed_soft','reed_hard'],
                   help="The channel matrix can be simple or complex. "
                    "simple is just a fixed stochastic matrix, "
                    "complex depend on the output of the last hidden layer of "
                    "the baseline model."
                    "reed_soft and reed_hard describe two different loss "
                    "used in Reed's paper.")
parser.add_argument('--trainable', action='store_false', default=True,
                   help="If False then use the best channel matrix for the "
                        "given permuation noise and do'nt train on it")
parser.add_argument('-v', '--verbose', action='store_true', default=False)
parser.add_argument('--pretrain', type=int, default=1,
                    help = "When using pretraining the baseline/simple model "
                    "is used as a start point for simple/complex model. "
                    "In addition a confusion matrix based on the  pretraining "
                    "model is used to initialize the bias of the channel matrix"
                    "0 = dont pretrain\n"
                    "1 = baseline as pretraining for simple.\n"
                    "2 = simple as pretraining for complex.\n"
                    "3 = as 2 but simple bias is start point for complex")
parser.add_argument('--pretrain_soft', action='store_true', default=False,
                   help="Use soft confusion matrix in pretraining")
parser.add_argument('--W', type=float, default=0,
                   help="the weightes of the channel matrix in the complex "
                    "method are initialzied uniform random number between "
                    "[-W/2,W/2]")
parser.add_argument('--batch_size', type=int, default=256,
                   help='reduce this if your GPU runs out of memory')
parser.add_argument('--nb_epoch', type=int, default=40,
                   help='increase this if you think the model does not overfit')
parser.add_argument('--patience', type=int, default=4,
                   help='Early stopping patience. Use 0 not to have early stopping.')
parser.add_argument('--stratify', action='store_false', default=True,
                   help="make sure every category of labels appears the same "
                    "number of times in training, noise, validation")
parser.add_argument('--noise_levels', type=float, nargs='*',
                   help="Noise levels to check.")
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['mnist', 'cifar100'],
                    help="What dataset to use.")
parser.add_argument('--sparse', type=int,
                    help="Use few baseline outputs when computing each "
                         "channel output. "
                         "The implementation shows the classification numerical"
                         " results but it does not show the run time improvement")

args = parser.parse_args()

FN = args.FN
print('Writining all results to %s...'%FN)

DOWN_SAMPLE = args.down_sample # what percentage of training data to use
# Comparing Jacob method to [Reed](http://arxiv.org/pdf/1412.6596v3.pdf)

# to make the experiments reproducable
seed = args.seed
np.random.seed(seed)  # for reproducibility
random.seed(seed)

# We train an MLP or CNN model to classify the labels
CNN=args.cnn

# The cross entropy loss of the output of the baseline MLP/CNN model is weighted
#  by `BETA`, for example if `BETA=1` we have a regular MLP/CNN model which is
#  also called the baseline model. In addition the output of the baseline model
#  is transformed through a "channel matrix" and a loss of the output of this
#  second output (we will call this `"channeled"`) is also measured using
#  cross entropy and weighted by `1-BETA`. You can have `BETA=0`
#  but this gives a degree of freedom to the output of the baseline to be
#  permuted by an additional unknown permutation (which will then be canceled
#  out by the channel matrix.) However, in our final measurement we want to see
#  how accurate the output of the baseline part of the combined model is and
#  therefore we dont want to have an unknown permuation. One way to help the
#  model avoid having an arbitrary permutation on the baseline is to have `BETA>0`
BETA=args.beta  # 1-BETA how much weight to give to the 2nd softmax loss and BETA for the standard/baseline 1st softmax

# The channel matrix can be simple or complex. Simple is just a fixed stochastic
#  matrix, complex depend on the output of the last hidden layer of the baseline model.
SIMPLE, COMPLEX, REED_SOFT, REED_HARD = range(4)
if args.model == 'simple':
    MODEL = SIMPLE
elif args.model == 'complex':
    MODEL = COMPLEX
elif args.model == 'reed_soft':
    MODEL = REED_SOFT
elif args.model == 'reed_hard':
    MODEL = REED_HARD
else:
    raise Exception('Unknown model type %s'%args.model)


# If False then use the best channel matrix for the given permuation noise and do'nt train on it
trainable=args.trainable
verbose=args.verbose
assert args.sparse is None or (trainable and MODEL in [SIMPLE,COMPLEX]),"sparse can only be used in trainable simple/complex"

# Run a baseline training and then use its labels to initialize the channel matrix for the full training
PRETRAIN = args.pretrain
PRETRAIN_SOFT = args.pretrain_soft

if BETA == 1 and PRETRAIN:
    print('you cant pretrain a baseline model')
    PRETRAIN = 0

if MODEL != SIMPLE:
    assert trainable==True,'you can use a fixed and non-trainable channel matrix only in SIMPLE'

# build a string which will identify the current experiment
if BETA==1:
    experiment = 'B'
elif MODEL == REED_SOFT:
    experiment = 'R'
elif MODEL == REED_HARD:
    experiment = 'r'
elif MODEL == SIMPLE:
    experiment = 'S' if trainable else 's'
elif MODEL == COMPLEX:
    experiment = 'C'
else:
    raise Exception('unknown model')
experiment += 'C' if CNN else 'M'
if BETA < 1:
    experiment += ('%g'%BETA)[2:]
if PRETRAIN:
    if PRETRAIN_SOFT:
        experiment += ['P', 'Q'][PRETRAIN - 1]  # r is not allowed
    else:
        experiment += ['p','q','r'][PRETRAIN-1]
    if args.sparse is not None:
        experiment += '%dS' % args.sparse

            # the weightes of the channel matrix in the complex method are initialzied
# uniform random number between [-W/2,W/2]
W=args.W # channel matrix weight initialization
if BETA < 1 and MODEL == COMPLEX:
    if W!=0.1:
        experiment += '%d'%(10*W)
experiment += '_%d' % seed

# baseline hyper parameters
batch_size = args.batch_size  # reduce this if your GPU runs out of memory

if args.dataset=='mnist':
    nb_classes = 10 # number of categories we classify. MNIST is 10 digits
    # input image dimensions. In CNN we think we have a "color" image with 1 channel of color.
    # in MLP with flatten the pixels to img_rows*img_cols
    img_color, img_rows, img_cols = 1, 28, 28
elif args.dataset=='cifar100':
    nb_classes = 100
    img_color, img_rows, img_cols = 3, 32, 32
img_size = img_color*img_rows*img_cols
if CNN:
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    nhiddens = [512]
    opt = 'adam' # SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    DROPOUT=0.5
    weight_decay = None
else:
    nhiddens = [500, 300]
    DROPOUT=0.5
    weight_decay = None # 1e-4
    opt='adam'

# We train with some of the training labels permuted by a fixed permutation
# OR generate a random permutation (change seed to have something different) or
# use a cyclic permutation
# # Repeat training with noise
if args.perm.startswith('random'):
    if args.perm == 'random':
        np.random.seed(seed)  # for reproducibility
        random.seed(seed)
    else:
        perm_seed = int(args.perm[len('random'):])
        np.random.seed(perm_seed)  # for reproducibility
        random.seed(perm_seed)

    # find a permutation with no stationary points
    while True:
        perm = np.random.permutation(nb_classes)
        if np.all(perm != np.arange(nb_classes)):
            break

    np.random.seed(seed)  # for reproducibility
    random.seed(seed)

elif args.perm == 'cyclic':
    perm = np.array([1,2,3,4,5,6,7,8,9,0])
elif args.perm == 'reed':
    # noise permutation: use this permutation (from Reed)
    perm = np.array([7, 9, 0, 4, 2, 1, 3, 5, 6, 8])  # noise permutation
else:
    perm = args.perm

# baseline model. We use the `Sequential` model from keras
# [cnn example](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py)
#  and keras [mlp example](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py)
#  as a single layer which computes the last hidden layer which we then use to
#  compute the baseline and as an input to the channel matrix
# the number of labels is adjusted to the data
regularizer = l2(weight_decay) if weight_decay else None

if isinstance(perm, basestring) and perm in ['weak','weak_hard','strong']:
    weak_model = Sequential(name='weak')

    if perm == 'strong':
        if CNN:
            weak_model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                            border_mode='valid',
                                            input_shape=(img_color, img_rows, img_cols)))
            weak_model.add(Activation('relu'))
            weak_model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
            weak_model.add(Activation('relu'))
            weak_model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
            weak_model.add(Dropout(0.25))

            weak_model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv, border_mode='same'))
            weak_model.add(Activation('relu'))
            weak_model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
            weak_model.add(Activation('relu'))
            weak_model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
            weak_model.add(Dropout(0.25))

            weak_model.add(Flatten())
            for nhidden in nhiddens:
                weak_model.add(Dense(nhidden, W_regularizer=regularizer))
                weak_model.add(Activation('relu'))
                weak_model.add(Dropout(DROPOUT))
        else:
            for i, nhidden in enumerate(nhiddens):
                weak_model.add(Dense(nhidden,
                                     input_shape=(img_size,) if i == 0 else [],
                                     W_regularizer=regularizer))
                weak_model.add(Activation('relu'))
                weak_model.add(Dropout(DROPOUT))

    weak_model.add(Dense(nb_classes, activation='softmax',
                         name='weak_dense',
                         input_shape=(img_size,)))
    weak_model.compile(loss='categorical_crossentropy', optimizer=opt)
    fname_weak_random_weights = '%s.%s.%s_model.hdf5' % (FN, experiment,perm)
    weak_model.save_weights(fname_weak_random_weights, overwrite=True)

hidden_layers = Sequential(name='hidden')
if CNN:
    hidden_layers.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(img_color, img_rows, img_cols)))
    hidden_layers.add(Activation('relu'))
    hidden_layers.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    hidden_layers.add(Activation('relu'))
    hidden_layers.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    hidden_layers.add(Dropout(0.25))

    hidden_layers.add(Convolution2D(nb_filters*2, 3, 3, border_mode='same'))
    hidden_layers.add(Activation('relu'))
    hidden_layers.add(Convolution2D(nb_filters*2, 3, 3))
    hidden_layers.add(Activation('relu'))
    hidden_layers.add(MaxPooling2D(pool_size=(2, 2)))
    hidden_layers.add(Dropout(0.25))

    hidden_layers.add(Flatten())
    for nhidden in nhiddens:
        hidden_layers.add(Dense(nhidden, W_regularizer=regularizer))
        hidden_layers.add(Activation('relu'))
        hidden_layers.add(Dropout(DROPOUT))
else:
    for i, nhidden in enumerate(nhiddens):
        hidden_layers.add(Dense(nhidden,
                                input_shape=(img_size,) if i == 0 else [],
                                W_regularizer=regularizer))
        hidden_layers.add(Activation('relu'))
        hidden_layers.add(Dropout(DROPOUT))

APRIOR_NOISE=0.46
if trainable:
    bias_weights = (
        np.array([np.array([np.log(1. - APRIOR_NOISE)
                            if i == j else
                            np.log(APRIOR_NOISE / (nb_classes - 1.))
                            for j in range(nb_classes)]) for i in
                  range(nb_classes)])
        + 0.01 * np.random.random((nb_classes, nb_classes)))
else:
    # use the ideal bias
    if isinstance(perm, np.ndarray):
        bias_weights = np.array([np.array([np.log(1.-APRIOR_NOISE)
                                                if i == j else
                                                (np.log(APRIOR_NOISE) if j == perm[i] else -1e8)
                                                for j in range(nb_classes)])
                                      for i in range(nb_classes)])
    else:
        bias_weights = np.array([np.array([np.log(1. - APRIOR_NOISE)
                                           if i == j else
                                           np.log(APRIOR_NOISE)/(nb_classes-1.)
                                           for j in range(nb_classes)])
                                 for i in range(nb_classes)])
inputs = Input(shape=(img_color,img_rows,img_cols) if CNN else (img_size,))
if MODEL == SIMPLE:
    # we need an input of constant=1 to derive the simple channel matrix from a regular softmax dense layer
    ones = Input(shape=(1,))

last_hidden = hidden_layers(inputs)

baseline_output = Dense(nb_classes, activation='softmax', name='baseline', W_regularizer=regularizer)(last_hidden)

if args.sparse is not None:
    class SparseMaskDense(Dense):
        """Keep a non trainable weights that should be either 1 or 0 for
        each of the outputs. When 0 use a very negative fixed bias to suppress
        that output."""
        def build(self, input_shape):
            super(SparseMaskDense, self).build(input_shape)
            self.sparse_mask = K.zeros((self.output_dim,),
                                     name='{}_sparse_mask'.format(self.name))
            self.non_trainable_weights = [self.sparse_mask]

        def call(self, x, mask=None):
            output = K.dot(x, self.W)
            if self.bias:
                output += self.b
            output = K.switch(self.sparse_mask, output, -1e20)
            return self.activation(output)

    channel_dense = SparseMaskDense
else:
    channel_dense = Dense


if MODEL == REED_SOFT or MODEL == REED_HARD:
    channeled_output = baseline_output
else:
    if MODEL == SIMPLE:
        # use bias=False and ones[:,:1] (and not bias=True and zeros) because we
        #  dont really need both bias and weights and there is no simple way to
        #  throwaway the weights
        channel_matrix = [channel_dense(nb_classes,
                                        activation='softmax',
                                        bias=False,
                                        name='dense_class%d'%i,
                                        trainable=trainable,
                                        weights=[
                                            bias_weights[i].reshape((1,-1))
                                        ])(ones)
                          for i in range(nb_classes)]
    elif MODEL == COMPLEX:
        channel_matrix = [channel_dense(nb_classes,
                                        activation='softmax',
                                        name='dense_class%d'%i,
                                        weights=[
                                            W*(np.random.random((nhidden,nb_classes)) - 0.5),
                                            bias_weights[i]
                                        ])(last_hidden)
                          for i in range(nb_classes)]
    channel_matrix = merge(channel_matrix, mode='concat')
    channel_matrix = Reshape((nb_classes,nb_classes))(channel_matrix)

    # multiply the channel matrix with the baseline output
    # channel_matrix.shape == (batch_size, nb_classes, nb_classes) and channel_matrix.sum(axis=-1) == 1
    # channel_matrix[b,0,0] is the probability that baseline output 0 will get to channeled_output 0
    # channel_matrix[b,0,1] is the probability that baseline output 0 will get to channeled_output 1 ...
    # ...
    # channel_matrix[b,1,0] is the probability that baseline output 1 will get to channeled_output 0 ...
    
    # baseline_output.shape == (batch_size, nb_classes) and baseline_output.sum(axis=-1) == 1
    # we want channeled_output[b,0] = channel_matrix[b,0,0] * baseline_output[b,0] + \
    #                                 channel_matrix[b,1,0] * baseline_output[b,1] + ...
    # so we do a dot product of axis 1 in channel_matrix with axis 1 in baseline_output
    channeled_output = merge([channel_matrix, baseline_output], mode='dot', dot_axes=(1,1), name='channeled')

def reed_soft_loss(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return -K.batch_dot(y_pred, K.log(y_pred+1e-8), axes=(1,1))

def reed_hard_loss(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return -K.log(K.max(y_pred, axis=1, keepdims=True)+1e-8)

if MODEL == REED_SOFT:
    loss = reed_soft_loss
elif MODEL == REED_HARD:
    loss = reed_hard_loss    
else:
    loss = 'categorical_crossentropy'

train_inputs = [inputs,ones] if MODEL == SIMPLE else inputs
if BETA==1:
    model = Model(input=train_inputs, output=baseline_output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
else:
    model = Model(input=train_inputs, output=[channeled_output, baseline_output])
    model.compile(loss=[loss, 'categorical_crossentropy'],loss_weights=[1.-BETA,BETA],
                  optimizer=opt,
                  metrics=['accuracy'])


# save the weights so we can latter re-load them every time we want to restart
#  training with a different noise level
fname_random_weights = '%s.%s.random.hdf5' % (FN, experiment)
model.save_weights(fname_random_weights, overwrite=True)

# Data:
# keras has a built in tool that download the MNIST data set for you to `~/.keras/datasets/`
# the data, shuffled and split between train and test sets
if args.dataset == 'mnist':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print('MNIST training data set label distribution', np.bincount(y_train))
    print('test distribution', np.bincount(y_test))
else:
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
    y_train = y_train.ravel()
    y_test = y_test.ravel()

STRATIFY=args.stratify
if STRATIFY:
    # make sure every category (of the nb_classes categories) appears the same number of times (N) in training
    # N is the size of the smallest category
    N = np.bincount(y_train).min()
    if DOWN_SAMPLE < 1:
        N = min(N*nb_classes, int(len(y_train) * DOWN_SAMPLE))
        idx, _ = next(iter(StratifiedShuffleSplit(n_splits=1,
                                                  train_size=N,
                                                  test_size=None,
                                                  random_state=seed).split(X_train,y_train)))
        X_train = X_train[idx]
        y_train = y_train[idx]
    print('stratified train', np.bincount(y_train))

else:
    N = len(X_train)
    idx = np.random.choice(N, min(int(N * DOWN_SAMPLE), N), replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]
    if DOWN_SAMPLE < 1:
        print('label distribution after downsampling', np.bincount(y_train))

if CNN:
    X_train = X_train.reshape(X_train.shape[0], img_color, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_color, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_size)
    X_test = X_test.reshape(X_test.shape[0], img_size)
    
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

if MODEL == SIMPLE:
    Ones_train = np.ones((len(X_train),1))
    Ones_test = np.ones((len(X_test),1))
    
    train_input = [X_train, Ones_train]
    test_input = [X_test, Ones_test]
else:
    train_input = X_train
    test_input = X_test

def fix_input(X):
    if MODEL == SIMPLE:
        # we generate the channel matrix by using softmax on dense on a constant
        # input of 1.
        ones = np.ones((len(X),1))
        return [X, ones]
    else:
        return X

def fix_output(y):
    # convert class vectors to binary class matrices
    Y = np_utils.to_categorical(y, nb_classes) if y.ndim == 1 else y
    if BETA==1:
        return Y
    else:
        return [Y, Y]

if isinstance(perm, basestring):
    experiment += '-' + perm
else:
    if nb_classes <= 10:
        experiment += '-'+''.join(map(str,perm))
    else:
        experiment += '-' + args.perm

experiment += '_%g'%(DOWN_SAMPLE*10)
if STRATIFY:
    experiment += 's'
print('Experiment', experiment)

if isinstance(perm, basestring) and perm in ['noise']:
    noise = np.mod(y_train + np.random.randint(1,nb_classes, y_train.shape),
                   nb_classes)
elif isinstance(perm, basestring) and perm in ['strong', 'weak', 'weak_hard']:
    noise = None
elif isinstance(perm, np.ndarray):
    noise = perm[y_train]
else:
    raise Exception('unknown perm %s'%perm)

model.load_weights(fname_random_weights)

def eval(model):
    return dict(zip(model.metrics_names,model.evaluate(fix_input(X_test),
                                                       fix_output(y_test),
                                                       verbose=False)))
print('Random classification', eval(model))

if args.noise_levels is not None:
    noise_levels = args.noise_levels
elif isinstance(perm, basestring) and perm == 'noise':
    noise_levels = np.array([0.2, 0.6,  0.7, 0.75, 0.8,  0.82,  0.84, 0.86,
                             0.88,  0.9, 0.92, 0.95, 1 ])
    # make sure you have enough (>nb_classes) labels (noised and not noised)
    noise_levels = np.clip(noise_levels, 0.01, 0.99)
elif isinstance(perm, basestring) and perm in ['weak','weak_hard','strong']:
    # In weak the noise level is the number of training samples we use
    # to train the weak classifier
    noise_levels = np.array([50, 100, 300, 1000, 3000, 5000])
else:
    noise_levels = np.array([0.3, 0.36,  0.38,  0.4 ,  0.42,  0.44,  0.46, 0.47,
                         0.475, 0.48, 0.485, 0.49, 0.495, 0.5 ])
    # make sure you have enough (>nb_classes) labels (noised and not noised)
    noise_levels = np.clip(noise_levels, 0.01, 0.99)

# repeat experiment for different noise level (percentage of training labels
# that are permutated)
for noise_level in tqdm(noise_levels):
    np.random.seed(seed)  # for reproducibility
    random.seed(seed)

    # replace some of the training labels with permuted (noise) labels.
    if noise_level <= 0:
        noise_idx = []
    elif STRATIFY:
        # make sure each categories receive an equal amount of noise
        _, noise_idx = next(iter(StratifiedShuffleSplit(n_splits=1,
                                                        test_size=noise_level,
                                                        random_state=seed).split(X_train,y_train)))
    else:
        N = len(y_train)
        if noise_level <= 1:
            noise_idx = np.random.choice(N, int(N * noise_level), replace=False)
        else:
            noise_idx = np.random.choice(N, int(noise_level), replace=False)

    if isinstance(perm, basestring) and perm in ['weak','weak_hard','strong']:
        weak_model.load_weights(fname_weak_random_weights)
        weak_model.fit(X_train[noise_idx],
                       np_utils.to_categorical(y_train[noise_idx], nb_classes),
                       batch_size=batch_size, nb_epoch=25, verbose=verbose)
        y_train_noise = weak_model.predict(X_train, batch_size=batch_size,
                                           verbose=verbose)
        y_train_noise_peak = np.argmax(y_train_noise, axis=-1)
        if perm in ['weak_hard']:
            y_train_noise = y_train_noise_peak
    else:
        y_train_noise = y_train.copy()
        y_train_noise[noise_idx] = noise[noise_idx]
        y_train_noise_peak = y_train_noise

    print('NOISE: level %.4f error %.4f' % (noise_level,
                                         1. - np.mean(
                                             y_train_noise_peak == y_train)))

    # always reset the entire model
    for t in range(5):
        try:
            model.load_weights(fname_random_weights)
            break
        except:
            print('FAILED TO LOAD RANDOM %s' % fname_random_weights)
            print('Trying again in 10sec')
            time.sleep(10)

    if PRETRAIN:
        #  start training with the best baseline model we have for the same
        #  noise (permutation and level of noise)

        # take the experiment name and convert it to either baseline (B + M/C)
        # or simple with hard pretraining=1 (S + M/C + p)
        # keepig all other parts of the experiment the same
        exparts = experiment.split('-')
        exparts0 = exparts[0].split('_')
        exparts00 = exparts0[0]
        # ignore the current pretrain mode and W
        if PRETRAIN > 1:
            exparts00 = 'S'+exparts00[1]+'p'
            # keep the same sparse when looking for a simple pre-training model
            # for a complex model
            if args.sparse is not None:
                exparts00 += '%dS' % args.sparse
        else:
            exparts00 = 'B'+exparts00[1]

        exparts0 = '_'.join([exparts00]+exparts0[1:])
        baseline_experiment = '-'.join([exparts0]+exparts[1:])

        lookup = {}
        ignore = []

        if PRETRAIN > 1:
            for i in range(nb_classes):
                k = 'dense_class%d_W'%i
                lookup[k] = 'dense_class%d_b'%i
                ignore.append(k)

        pretrained_model_name = '%s.%s.%f.hdf5'%(FN,baseline_experiment,noise_level)

        for t in range(5):
            try:
                with fasteners.InterProcessLock('/tmp/%s.lock_file'%FN):
                    # model.load_weights(pretrained_model_name, by_name=True)
                    # same as running model.load_weights(..., by_name=True)
                    load_weights(model, pretrained_model_name,
                                 lookup=lookup,
                                 ignore=ignore)
                break
            except:
                print('FAILED TO LOAD BASELINE %s'%pretrained_model_name)
                print('Trying again in 10sec')
                time.sleep(10)
        else:
            raise Exception('ABORTING because the baseline model file was not found'
                            'consider re-running with --beta=1')
        if verbose:
            print('Baseline classification', eval(model))
        if trainable:
            if MODEL in [SIMPLE, COMPLEX] and PRETRAIN < 3:
                # build confusion matrix (prediction,noisy_label)
                ybaseline_predict = model.predict(fix_input(X_train),
                                                  batch_size=batch_size)[1]
                perm_bias_weights = np.zeros((nb_classes, nb_classes))
                if PRETRAIN_SOFT:
                    for n, p in zip(y_train_noise, ybaseline_predict):
                        perm_bias_weights[:, n] += p
                else:
                    ybaseline_predict = np.argmax(ybaseline_predict, axis=-1)
                    if y_train_noise.ndim == 1:
                        for n, p in zip(y_train_noise, ybaseline_predict):
                            perm_bias_weights[p, n] += 1.
                    else:
                        for n, p in zip(y_train_noise, ybaseline_predict):
                            perm_bias_weights[p, :] += n
                if args.sparse is not None:
                    # for each output from the baseline model, keep track
                    # which outputs we want it to affect.
                    sparse_mask = np.ones((nb_classes, nb_classes))
                    # start with the confusion matrix built from the base model
                    # and for each baseline output find the top outputs
                    channel_input_idx = perm_bias_weights.argsort()[:,::-1]
                    for i in range(nb_classes):
                        # keep the top args.sparse set to one and all others to zero
                        sparse_mask[i, channel_input_idx[i, args.sparse:]] = 0.
                    # zero also the matching places in the confusion matrix
                    perm_bias_weights = perm_bias_weights * sparse_mask
                perm_bias_weights /= perm_bias_weights.sum(axis=1, keepdims=True)
                # perm_bias_weights[prediction,noisy_label] = log(P(noisy_label|prediction))
                perm_bias_weights = np.log(perm_bias_weights + 1e-8)
                for i in range(nb_classes):
                    if MODEL == SIMPLE:
                        #  given we predict <i> in the baseline model,
                        #  dense_class<i> gives log(P(noisy_label))
                        K.set_value(model.get_layer(name='dense_class%d'%i).trainable_weights[0],
                                    perm_bias_weights[i].reshape((1,-1)))
                    else:
                        K.set_value(model.get_layer(name='dense_class%d'%i).trainable_weights[1],
                                    perm_bias_weights[i])
                    if args.sparse is not None:
                        K.set_value(model.get_layer(name='dense_class%d'%i).non_trainable_weights[0],
                                    sparse_mask[i])

        else:
            def calc_perm_bias_weights(noise_level):
                if isinstance(perm, np.ndarray):
                    perm_bias_weights = np.array(
                        [np.array([np.log(1. - noise_level) if i == j else
                                   (np.log(noise_level) if j == perm[
                                       i] else  # experiment = s...
                                    -1e8)
                                   for j in range(nb_classes)]) for i in
                         range(nb_classes)])
                else:
                    perm_bias_weights = np.array(
                        [np.array([np.log(1. - noise_level) if i == j else
                                   np.log(noise_level) / (nb_classes - 1.)
                                   for j in range(nb_classes)]) for i in
                         range(nb_classes)])
                return perm_bias_weights

            perm_bias_weights = calc_perm_bias_weights(noise_level)
            for i in range(nb_classes):
                K.set_value(model.get_layer(name='dense_class%d'%i).non_trainable_weights[0],
                            perm_bias_weights[i].reshape((1,-1)))

    # break the training set to 10% validation which we will use for early stopping.
    if STRATIFY:
        train_idx, val_idx = next(iter(
                StratifiedShuffleSplit(n_splits=1, test_size=0.1,
                                       random_state=seed).split(X_train, y_train_noise_peak)))
        X_train_train = X_train[train_idx]
        y_train_train = y_train_noise[train_idx]
        X_train_val = X_train[val_idx]
        y_train_val = y_train_noise[val_idx]
    else:
        Nv = len(X_train) // 10
        X_train_train = X_train[Nv:]
        y_train_train = y_train_noise[Nv:]
        X_train_val = X_train[:Nv]
        y_train_val = y_train_noise[:Nv]

    train_res = model.fit(fix_input(X_train_train),
                          fix_output(y_train_train),
                          batch_size=batch_size,
                          nb_epoch=args.nb_epoch,
                          verbose=verbose,
                          validation_data=(fix_input(X_train_val),
                                           fix_output(y_train_val)),
                          callbacks=
                          [EarlyStopping(patience=args.patience,mode='min',
                                         verbose=verbose)]
                          if args.patience > 0 else []
                          )

    eval_res = eval(model)
    print('End classification', eval_res)
    # lock all operations on results pkl
    # so we can run multiple experiments at the same time
    with fasteners.InterProcessLock('/tmp/%s.lock_file'%FN):
        try:
            with open('%s.results.pkl'%FN,'rb') as fp:
                results = pickle.load(fp)
        except:
            results = {}
        results[(experiment,noise_level)] = (train_res.history,eval_res)
        with open('%s.results.pkl'%FN,'wb') as fp:
            pickle.dump(results, fp, -1)

    # save model
    model.save_weights('%s.%s.%f.hdf5'%(FN,experiment,noise_level),
                       overwrite=True)
