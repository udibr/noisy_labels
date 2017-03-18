# noisy_labels
<a target="_blank" href="http://twitter.com/udibr"><img alt='Twitter followers' src="https://img.shields.io/twitter/follow/udibr.svg?style=social"></a>

TRAINING DEEP NEURAL-NETWORKS USING A NOISE ADAPTATION LAYER
[ICLR 2017 conference submission](http://openreview.net/forum?id=H12GRgcxg)

Learning MNIST when almost half the labels are permuted in a fixed way. For example, when the task of labeling is split between two people that don’t agree.

Follow [mnist-simple](./mnist-simple.ipynb) notebook for an example of how to implement the Simple noise adaption layer in the paper with a single customized Keras layer.
Follow [161103-run-plot](./161103-run-plot.ipynb), [161202-run-plot-cifar100](./161202-run-plot-cifar100.ipynb) and [161230-run-plot-cifar100-sparse](./161230-run-plot-cifar100-sparse.ipynb) notebooks for how to reproduce the results of the paper.
