import tensorflow as tf
from tensorflow.keras import layers
import copy
import utils


def ConvLSTM(in_shape, name):
    inputs = layers.Input(in_shape)
    x = layers.ConvLSTM2D(filters=8, kernel_size=4, strides=4)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPool2D(name='global_average_pooling')(x)  # Global Average Pooling act as dropout; so no need to add a dropout layers after
    return tf.keras.Model(inputs, x, name=name)


def LSTM(in_shape, name):
    inputs = layers.Input(in_shape)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name='global_average_pooling')(inputs)
    x = layers.LSTM(units=50)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # for the following fully-connected layers
    return tf.keras.Model(inputs, x, name=name)


def ConvNet1D(in_shape, name):
    def temporal_convolution_block(block, filters, kernel_size, inputs):
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='VALID', name='{}_temporal_conv_1'.format(block))(inputs)
        x = layers.TimeDistributed(layers.BatchNormalization(), name='{}_bn_1'.format(block))(x)
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='VALID', name='{}_temporal_conv_2'.format(block))(x)
        x = layers.TimeDistributed(layers.BatchNormalization(), name='{}_bn_2'.format(block))(x)
        out = layers.TimeDistributed(layers.LeakyReLU(alpha=0.01), name='{}_leaky_relu'.format(block))(x)
        return out
    
    inputs = layers.Input(in_shape)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name='global_avg_pool')(inputs)
    x = temporal_convolution_block('block_1', 10, 3, x)
    x = temporal_convolution_block('block_2', 10, 3, x)
    x = temporal_convolution_block('block_3', 10, 3, x)
    x = temporal_convolution_block('block_4', 10, 3, x)
    x = layers.Conv1D(filters=100, kernel_size=4, padding='VALID', name='temporal_conv')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name='batch_norm')(x)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.01), name='out_leaky_relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)  # for the following fully-connected layers
    return tf.keras.Model(inputs, x, name=name)


class Model(tf.keras.Model):
    def __init__(self, _name, _type, _verb_only=False, out_verbs=97, out_nouns=300):
        """ Construct a model with Convolutional layers and stacked ConvLSTM layers. """
        
        assert _type in ['ConvLSTM', 'LSTM', 'ConvNet1D'], 'type of model must be one of: ConvNet1D | LSTM | ConvLSTM'
        super(Model, self).__init__(name=_name)
        self.verb_only = _verb_only
        self.get_feature_maps = False
        
        # Backbone
        self.backbone = utils.get_TimeDistributed_MobilenetV2('mobilenet')
        self.backbone(layers.Input((None, 256, 256, 3)))  # call the layer to create the output shape
        
        # Time flow
        if _type == 'ConvLSTM':
            self.time_flow = ConvLSTM(self.backbone.output_shape[1:], 'conv_lstm')
        elif _type == 'LSTM':
            self.time_flow = LSTM(self.backbone.output_shape[1:], 'lstm')
        elif _type == 'ConvNet1D':
            self.time_flow = ConvNet1D(self.backbone.output_shape[1:], 'temporal_conv')

        # Classification
        inputs = layers.Input(self.time_flow.output_shape[1:])
        if self.verb_only:
            outputs = layers.Dense(out_verbs, activation='softmax', name='verb', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))(inputs)
        else:
            verb = layers.Dense(out_verbs, activation='softmax', name='verb', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))(inputs)
            noun = layers.Dense(out_nouns, activation='softmax', name='noun', kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))(inputs)
            outputs = [verb, noun]
        self.classifier = tf.keras.Model(inputs, outputs, name='fully_connected')
    
    def call(self, inputs):
        """
            Customize what happens in call function so that I can extract feature maps for activations analysis.
            Cannot use the 'training' argument to know if training/inference mode because in validation step it is set to 'False';
                which means that it will return the feature maps in a validation step in fit().
                However I define 'self.get_feature_maps' in this purpose, and set it to 'True' externally when feature maps are needed
            
            see "test_step(self, data)" function in tf.keras.Model (https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/engine/training.py#L1242-L1394) l.1148

        """
        backbone_feature_maps = self.backbone(inputs)
        time_feature_maps = self.time_flow(backbone_feature_maps)
        logits = self.classifier(time_feature_maps)
        if self.get_feature_maps:
            return backbone_feature_maps, time_feature_maps, logits
        else:
            return logits
