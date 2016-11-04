import numpy as np
import h5py
import keras.backend as K

def str_shape(x):
    return 'x'.join(map(str, x.shape))


def load_weights(model, filepath, lookup={}, ignore=[], transform=None, verbose=True):
    """Modified version of keras load_weights that loads as much as it can
    Useful for transfer learning.

    read the weights of layers stored in file and copy them to a model layer.
    the name of each layer is used to match the file's layers with the model's.
    It is possible to have layers in the model that dont appear in the file..

    The loading stopps if a problem is encountered and the weights of the
    file layer that first caused the problem are returned.

    # Arguments
        model: Model
            target
        filepath: str
            source hdf5 file
        lookup: dict (optional)
            by default, the weights of each layer in the file are copied to the
            layer with the same name in the model. Using lookup you can replace
            the file name with a different model layer name, or to a list of
            model layer names, in which case the same weights will be copied
            to all layer models.
        ignore: list (optional)
            list of model layer names to ignore in
        transform: None (optional)
            This is an optional function that receives the list of weighs
            read from a layer in the file and the model layer object to which
            these weights should be loaded.
        verbose: bool
            high recommended to keep this true and to follow the print messages.

    # Returns
        weights of the file layer which first caused the load to abort or None
        on successful load.

    """
    if verbose:
        print 'Loading', filepath, 'to', model.name
    flattened_layers = model.layers
    with h5py.File(filepath, mode='r') as f:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for name in layer_names:
            if verbose:
                print name,
            g = f[name]
            weight_names = [n.decode('utf8') for n in
                            g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in
                                 weight_names]
                if verbose:
                    print 'loading', ' '.join(
                            str_shape(w) for w in weight_values),
                target_names = lookup.get(name, name)
                if isinstance(target_names, basestring):
                    target_names = [target_names]
                # handle the case were lookup asks to send the same weight to multiple layers
                target_names = [target_name for target_name in target_names if
                                target_name == name or target_name not in layer_names]
                for target_name in target_names:
                    if verbose:
                        print target_name,
                    try:
                        layer = model.get_layer(name=target_name)
                    except:
                        layer = None
                    if layer:
                        # the same weight_values are copied to each of the target layers
                        symbolic_weights = layer.trainable_weights + layer.non_trainable_weights

                        if transform is not None:
                            transformed_weight_values = transform(weight_values, layer)
                            if transformed_weight_values is not None:
                                if verbose:
                                    print '(%d->%d)'%(len(weight_values),len(transformed_weight_values)),
                                weight_values = transformed_weight_values

                        problem = False
                        weight_values_names = [w.name.split('/')[-1] for w in weight_values]
                        weight_values_names = [lookup.get(w,w) for w in weight_values_names]
                        weight_pairs = []
                        for symbolic_weight in symbolic_weights:
                            symbolic_weight_name = symbolic_weight.name
                            if symbolic_weight_name in weight_values_names:
                                idx = weight_values_names.index(symbolic_weight_name)
                                if symbolic_weight_name in weight_values_names[idx+1:]:
                                    problem = True
                                    print symbolic_weight_name, 'appeared more than once',
                                else:
                                    weight_value = weight_values[idx]
                                    old_weight_value = K.get_value(symbolic_weight)
                                    if old_weight_value.dtype != weight_value.dtype:
                                        problem = True
                                        print symbolic_weight_name,'type chnaged from',old_weight_value.dtype,'to',weight_value.dtype,
                                    if old_weight_value.shape != weight_value.shape:
                                        print symbolic_weight_name,'shape chnaged from',old_weight_value.shape,'to',weight_value.shape,
                                        if np.prod(old_weight_value.shape) == np.prod(weight_value.shape):
                                            print 'reshaping new value to old shape',
                                            weight_pairs.append(
                                                (symbolic_weight,
                                                 np.reshape(weight_value,old_weight_value.shape)))
                                        else:
                                            problem = True
                                    else:
                                        weight_pairs.append((symbolic_weight, weight_value))
                            else:
                                print symbolic_weight_name, 'is missing',
                                if ignore != '*' and symbolic_weight_name not in ignore:
                                    problem = True
                        if problem and verbose:
                            print '(bad #wgts)',
                        if not problem:
                            weight_value_tuples += weight_pairs
                    else:
                        problem = True
                    if problem:
                        if verbose:
                            if name in ignore or ignore == '*':
                                print '(skipping)',
                            else:
                                print 'ABORT'
                        if not (name in ignore or ignore == '*'):
                            K.batch_set_value(weight_value_tuples)
                            return [np.array(w) for w in weight_values]
                if verbose:
                    print
            else:
                if verbose:
                    print 'skipping this is empty file layer'
        K.batch_set_value(weight_value_tuples)
