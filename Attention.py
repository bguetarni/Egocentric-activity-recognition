import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import utils
import OpticalFlow


def Attention(name, k=20):
    """ Attention mechanism using optical flows computed by a pre-trained model """
    keys = layers.Input((k-1, 32, 32, 1))  # optical flows between gray images -> if there are k RGB images then there will be k-1 optical flow fields
    values = layers.Input((k, 256, 256, 3))  # RGB images
    
    upsampler = layers.UpSampling2D(size=8, data_format='channels_last')
    # the UpSampling layer allow to replicate localally.
    # e.g. 2x2 UpSampling     12   -->      1122
    #                         34            1122
    #                                       3344
    #                                       3344
    # this allow to resize the attention weights to the image size for a simple dot product whil keeping the weights identical in a local zone
    
    y = []
    for t in range(values.shape[1]):
        # create weights associated with each RGB image as follow:
        #   - first image is weighted by optical flow computed between the first and second images
        #   - last image is weighted by optical flow computed between the last and before last images
        #   - any image between them is weighted by both optical flows computed with it (i.e. image t -> optical flows t-1 and t)
        if t == 0:
            y.append(upsampler(keys[:, t]))
        elif t == values.shape[1] - 1:
            y.append(upsampler(keys[:, t - 1]))
        else:
            y.append(upsampler(keys[:, t - 1]) + upsampler(keys[:, t]))
    w = tf.stack(y, axis=1)  # group the weights
    w = tf.clip_by_value(w, 0.5, 1., name='clipping')  # clip the weights in [0.5,1]
    out = tf.math.multiply(values, w)  # apply weights
    return tf.keras.Model([values, keys], out, name=name)


def TemporalCNN(name, shape):
    def block(block, filters, kernel_size, x):
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='VALID', name='{}_temporal_conv_1'.format(block))(x)
        x = layers.TimeDistributed(layers.BatchNormalization(), name='{}_bn_1'.format(block))(x)
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='VALID', name='{}_temporal_conv_2'.format(block))(x)
        x = layers.TimeDistributed(layers.BatchNormalization(), name='{}_bn_2'.format(block))(x)
        x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.01), name='{}_leaky_relu'.format(block))(x)
        return x

    inputs = layers.Input(shape)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name='global_avg_pool')(inputs)
    x = block('block_1', filters=64, kernel_size=3, x=x)
    x = block('block_2', filters=64, kernel_size=3, x=x)
    x = block('block_3', filters=64, kernel_size=3, x=x)
    x = block('block_4', filters=64, kernel_size=3, x=x)
    x = layers.Conv1D(filters=128, kernel_size=4, padding='VALID', activation='tanh', name='temporal_conv')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name='batch_norm')(x)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.01), name='leaky_relu')(x)
    return tf.keras.Model(inputs, x, name=name)


class SliceTensor(layers.Layer):
    """ Layer that creates pairs of RGB images """
    def __init__(self, name):
        super().__init__(name)
    
    def __call__(self, inputs):
        # inputs = (N, k, 256, 256, 3)
        out = []
        for i in range(inputs.shape[1] - 1):
            out.append(tf.concat([inputs[:, i], inputs[:, i+1]], axis=-1))
        return tf.stack(out, axis=1)  # -> (N, k-1, 256, 256, 6)


class AttentionTemporalCNN(tf.keras.Model):
    """ Model that uses optical flow as an attention mechanism """
    def __init__(self, name, k=20, verb_only=False, joint_learning=False, out_verbs=97, out_nouns=300):
        super().__init__(name)
        self.k = k
        self.verb_only = verb_only

        optflow = tf.keras.models.load_model('opticalflow.tf', compile=False)  # make sure that a pre-trained optical flow cnn is in the folder in the SavedModel format
        inputs = layers.Input((k, 256, 256, 3))
        x = SliceTensor('slicer')(inputs)
        x = layers.TimeDistributed(optflow)(x)
        self.opticalflow = tf.keras.Model(inputs, x, 'opticalflow')
        if not joint_learning:
            self.opticalflow.trainable = False

        self.attention = Attention('attention')
        self.mobilenet = utils.get_TimeDistributed_MobilenetV2('mobilenet')
        self.mobilenet.trainable = False
        self.temporal_conv = TemporalCNN('temporal_conv', self.mobilenet.output_shape[1:])

        inputs = layers.Input(self.temporal_conv.output_shape[1:])
        x = layers.Flatten()(inputs)
        x = layers.Dropout(0.5)(x)
        if self.verb_only:
            out = layers.Dense(units=13, activation='softmax', name='verb')(x)
        else:
            verb = layers.Dense(units=13, activation='softmax', name='verb')(x)
            noun = layers.Dense(units=21, activation='softmax', name='noun')(x)
            out = [verb, noun]
        self.dense = tf.keras.Model(inputs, out, name='fully_connected')
    
    def call(self, inputs, training=False):
        optical_flows = self.opticalflow(inputs)
        weighted = self.attention([inputs, optical_flows])
        feature_maps = self.mobilenet(weighted)
        temp_features = self.temporal_conv(feature_maps)
        out = self.dense(temp_features)
        if training:
            return out
        else:
            return out, temp_features, feature_maps, weighted, optical_flows
