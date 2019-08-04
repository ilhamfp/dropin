# Modified version of https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/core.py 
# and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

import inspect
import os

import collections
import numbers

import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.keras import backend as K

class Dropin(Layer):
    """Applies Dropin to the input.
    Dropin consists in randomly setting
    a fraction `rate` of input units to the corresponding value at each update during training time,
    which helps prevent overfitting.
    Arguments:
        rate: Float between 0 and 1. Fraction of the input units to drop in.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropin mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropin mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropin) or in inference mode (doing nothing).
    """

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropin, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        # Subclasses of `Dropin` may implement `_get_noise_shape(self, inputs)`,
        # which will override `self.noise_shape`, and allows for custom noise
        # shapes with dynamically sized inputs.
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            return dropin(inputs, 
                            self.rate, 
                            noise_shape=self._get_noise_shape(inputs), 
                            seed=self.seed)

        output = tf_utils.smart_cond(training,
                                     dropped_inputs,
                                     lambda: array_ops.identity(inputs))
        
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
                'rate': self.rate,
                'noise_shape': self.noise_shape,
                'seed': self.seed
        }
        base_config = super(Dropin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def dropin(x, rate, noise_shape=None, seed=None, name=None):
    """Computes dropin.
    With probability `rate`, drops in elements of `x`. Input that are dropped in are
    scaled up by `1 / (1 - rate)`, otherwise outputs `0`.    The scaling is so that
    the expected sum is unchanged.
    
    Args:
        x: A floating point tensor.
        rate: A scalar `Tensor` with the same type as x. The probability
            that each element is dropped. For example, setting rate=0.1 would drop
            10% of input elements.
        noise_shape: A 1-D `Tensor` of type `int32`, representing the
            shape for randomly generated keep/drop flags.
        seed: A Python integer. Used to create random seeds. See
            `tf.compat.v1.set_random_seed` for behavior.
        name: A name for this operation (optional).
    Returns:
        A Tensor of the same shape of `x`.
    Raises:
        ValueError: If `rate` is not in `(0, 1]` or if `x` is not a floating point
            tensor.
    """
    with ops.name_scope(name, "dropin", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                                             " be scaled. Got a %s tensor instead." % x.dtype)
        if isinstance(rate, numbers.Real):
            if not (rate >= 0 and rate < 1):
                raise ValueError("rate must be a scalar tensor or a float in the "
                                                 "range [0, 1), got %g" % rate)
            if rate < 0.5:
                logging.log_first_n(
                        logging.WARN, "Low dropin rate: %g (<0.5). In TensorFlow "
                        "2.x, dropin() uses dropin rate instead of keep_prob. "
                        "Please ensure that this is intended.", 5, rate)

        # Early return if nothing needs to be dropped.
        if isinstance(rate, numbers.Real) and rate == 0:
            return x
        if context.executing_eagerly():
            if isinstance(rate, ops.EagerTensor):
                if rate.numpy() == 0:
                    return x
        else:
            rate = ops.convert_to_tensor(
                    rate, dtype=x.dtype, name="rate")
            rate.get_shape().assert_is_compatible_with(tensor_shape.scalar())

            # Do nothing if we know rate == 0
            if tensor_util.constant_value(rate) == 0:
                return x

        noise_shape = _get_noise_shape(x, noise_shape)
        # Sample a uniform distribution on [0.0, 1.0) and select values larger than
        # rate.
        #
        # NOTE: Random uniform actually can only generate 2^23 floats on [1.0, 2.0)
        # and subtract 1.0.
        random_tensor = random_ops.random_uniform(
                noise_shape, seed=seed, dtype=x.dtype)

        scale = 1 / rate
        # NOTE: if (1.0 + rate) - 1 is equal to rate, then we want to consider that
        # float to be selected, hence we use a >= comparison.
        keep_mask = random_tensor < rate
        ret = x * scale * math_ops.cast(keep_mask, x.dtype)
        if not context.executing_eagerly():
            ret.set_shape(x.get_shape())
            
        return ret

def _get_noise_shape(x, noise_shape):
    # If noise_shape is none return immediately.
    if noise_shape is None:
        return array_ops.shape(x)

    try:
        # Best effort to figure out the intended shape.
        # If not possible, let the op to handle it.
        # In eager mode exception will show up.
        noise_shape_ = tensor_shape.as_shape(noise_shape)
    except (TypeError, ValueError):
        return noise_shape

    if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
        new_dims = []
        for i, dim in enumerate(x.shape.dims):
            if noise_shape_.dims[i].value is None and dim.value is not None:
                new_dims.append(dim.value)
            else:
                new_dims.append(noise_shape_.dims[i].value)
        return tensor_shape.TensorShape(new_dims)

    return noise_shape
