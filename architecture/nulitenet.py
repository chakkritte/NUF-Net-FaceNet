"""
NU-LiteNet.

As described in https://www.sciencedirect.com/science/article/abs/pii/S0952197619301824

  On-Device facial verification using NUF-Net model of deep learning

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def fire_module_res(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                activation_fn=tf.nn.relu,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)

            up = slim.conv2d(outputs, inputs.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
            inputs += up

            if activation_fn:
                inputs = activation_fn(inputs)
            return inputs

def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)
            return outputs

def squeeze(inputs, num_outputs):
    return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')

def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1 , padding='SAME' ,  scope='1x1')
		
        e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], padding='SAME', scope='3x3')
		
        e5x5 = slim.conv2d(inputs, num_outputs , [5, 5], padding='SAME', scope='5x5')

        e7x7 = slim.conv2d(inputs, num_outputs , [7, 7], padding='SAME', scope='7x7')
		
    return tf.concat([e1x1, e3x3, e5x5, e7x7], 3)

def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.00004, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }


    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.09),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        activation_fn=tf.nn.relu,
                       ):
        with tf.variable_scope('nulitenet', [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                net = slim.conv2d(images, 64, [7, 7], stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')  
                net = fire_module_res(net, 16, 16, scope='fire1') 
                net = fire_module_res(net, 16, 16, scope='fire2')
                net = fire_module_res(net, 32, 32, scope='fire3')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool2')
                net = fire_module_res(net, 32, 32, scope='fire4') 
                net = fire_module_res(net, 32, 32, scope='fire5')
                net = fire_module_res(net, 64, 64, scope='fire6')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool3')
                net = fire_module_res(net, 64, 64, scope='fire7')
                net = fire_module_res(net, 64, 128, scope='fire8')
                net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv10') #add new
                net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avgpool4')
                net = tf.squeeze(net, [1, 2], name='logits')
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=False)
    return net, None
