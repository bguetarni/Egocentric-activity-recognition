import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import math
import time
import cv2
import os
import argparse
import datetime
import utils


class CustomDataLoader():
    """ Dataset custom class for optical flows training """
    def __init__(self, batch_size, validation_split, in_path, out_path):
        """
            Create the dataset.
            Note: the images are stored as uint8 to occupy less space; we need to convert them before transforming to Tensor.

            Args:
                (int)   batch_size:         batch size to use during training
                (float) validation_split:   validation split (e.g. 0.2 -> validation set will take 20% of the dataset)
                (str)   in_path:            path to the RGB images
                (str)   out_path:           path to the optical flows
        """
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.idx = 0

        # create the dataset
        data = os.listdir(args.in_path)
        self.X = np.empty((len(data), 256, 256, 6), dtype='uint8')  # 2 RGB frames (input)
        self.Y = np.empty((len(data), 256, 256, 2), dtype='uint8')  # optical flows (u,v) (output)
        for i, sample in enumerate(data):
            print('\rLoading data: {}/{}      '.format(i+1, len(data)), end='')
            x, y = self.__getsample__(sample, in_path, out_path)
            self.X[i] = x
            self.Y[i] = y

    def __getsample__(self, sample, in_path, out_path):
        """ Return one random sample.
        Args:
            (str) in_path:  path to the RGB images
            (str) out_path: path to the optical flows

        in_path folder must be like:
            in_path\
                    0001\
                        rgb1.jpg
                        rgb2.jpg
                    0002\
                        0.jpg
                        1.jpg
                    .
                    .
                    .

        out_path folder must be like:
            out_path\
                    0001\
                        u.jpg
                        v.jpg
                    0002\
                        u.jpg
                        v.jpg
                    .
                    .
                    .
        """
        # In the 55 version, the optical flows are computed with:
        #   y(t) = TV-L1(x(2t), x(2t + 3))
        # In the 100 version:
        # see https://github.com/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM/blob/master/src/convert_rgb_to_flow_frame_idxs.py

        # read
        rgb_1 = cv2.imread(in_path + sample + '/rgb1.jpg')
        rgb_2 = cv2.imread(in_path + sample + '/rgb2.jpg')
        u = cv2.imread(out_path + sample + '/u.jpg')[:, :, 0]
        v = cv2.imread(out_path + sample + '/v.jpg')[:, :, 0]
        
        # resize
        rgb_1 = cv2.resize(rgb_1, (256, 256))
        rgb_2 = cv2.resize(rgb_2, (256, 256))
        u = cv2.resize(u, (256, 256))
        v = cv2.resize(v, (256, 256))
        
        # concatenate and stack
        rgb = np.concatenate((rgb_1, rgb_2), axis=-1)
        optf = np.stack((u, v), axis=-1)
        return rgb, optf
    
    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx + self.batch_size < self.X.shape[0]:
            indices = range(self.idx, self.idx + self.batch_size)
        elif self.idx < self.X.shape[0] - 1:
            indices = range(self.idx, self.X.shape[0])
        else:
            raise StopIteration
        
        # update index
        self.idx += self.batch_size
        
        # input
        inp = self.X[indices].astype('float32')
        
        # output
        #    Optical flow fields are in the range [0,255].
        #    A pixel equal to 255/2=127.5 has no motion.
        #    The farthest a pixel is from 127.5 the stronger the motion is.
        #    To convert the optical flow fields to magnitude we have to translate them into [0,1] with 0 equivalent to no motion (127.5), and 1 equivalent to strong motion (0 or 255).
        u = self.Y[indices][..., 0].astype('float32')/255.  # extract optical flow field 'u' and scale it
        u = u.reshape(u.shape + (1,))  # tf.Variable with shape (N,H,W,1)
        v = self.Y[indices][..., 1].astype('float32')/255.  # extract optical flow field 'v' and scale it
        v = v.reshape(v.shape + (1,))  # tf.Variable with shape (N,H,W,1)
        out = tf.math.sqrt(u**2 + v**2)  # optical flow magnitude
        out = out/tf.math.sqrt(2.)  # [0,sqrt(2)] -> [0,1]
        out = 2.*out - 1.  # [0,1] -> [-1,1]   TRANSLATION (step 1)
        out_1 = tf.math.exp(-0.006/(out**2 + 1e-5))
        out_2 = tf.math.sqrt(tf.math.abs(out)**0.6)
        out = (out_1 + out_2)/2.  # [-1,1] -> [0,1]        TRANSLATION (step 2)
        out = tf.nn.max_pool(out, ksize=8, strides=8, padding='VALID').numpy()  # (256,256) -> (32,32) using max value of 32x32 blocks

        if self.idx < math.floor(self.X.shape[0]*(1 - self.validation_split)):
            return (inp, out), True
        else:
            return (inp, out), False
    
    def number_of_batches(self):
        return math.ceil(self.X.shape[0]/self.batch_size)


def OpticalFlow(name):
    def block(blockID, x, filters, kernel_size, padding):
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, name='{}_conv_1'.format(blockID))(x)
        x = layers.BatchNormalization(name='{}_bn_1'.format(blockID))(x)
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, name='{}_conv_2'.format(blockID))(x)
        x = layers.BatchNormalization(name='{}_bn_2'.format(blockID))(x)
        x = layers.LeakyReLU(alpha=0.01, name='{}_leaky_relu'.format(blockID))(x)
        return x

    inputs = layers.Input(shape=(256, 256, 6))  # RGB images
    x = block('block_0', inputs, filters=2, kernel_size=5, padding='valid')  # -> (248,248,2)
    x = layers.MaxPool2D()(x)
    residual = x
    x = block('block_1', x, filters=2, kernel_size=3, padding='same')  # -> (124,124,2)
    x = layers.Concatenate()([x, residual])
    x = layers.MaxPool2D()(x)
    x = block('block_2', x, filters=2, kernel_size=5, padding='valid')  # -> (54,54,2)
    residual = x
    x = block('block_3', x, filters=2, kernel_size=3, padding='same')  # -> (54,54,2)
    x = layers.Concatenate()([x, residual])
    x = block('block_4', x, filters=2, kernel_size=5, padding='valid')  # -> (46,46,2)
    residual = x
    x = block('block_5', x, filters=2, kernel_size=3, padding='same')  # -> (46,46,2)
    x = layers.Concatenate()([x, residual])
    x = block('block_6', x, filters=2, kernel_size=3, padding='valid')  # -> (42,42,2)
    residual = x
    x = block('block_7', x, filters=2, kernel_size=3, padding='same')  # -> (42,42,2)
    x = layers.Concatenate()([x, residual])
    x = block('block_8', x, filters=2, kernel_size=3, padding='valid')  # -> (38,38,2)
    residual = x
    x = block('block_9', x, filters=2, kernel_size=3, padding='same')  # -> (38,38,2)
    x = layers.Concatenate()([x, residual])
    x = block('block_10', x, filters=2, kernel_size=3, padding='valid')  # -> (34,34,2)
    x = layers.Conv2D(filters=1, kernel_size=3, padding='valid', name='temporal_conv')(x)  # -> (32,32,1)
    x = layers.BatchNormalization(name='batch_norm')(x)
    x = tf.keras.layers.Activation('sigmoid', name='activation')(x)
    return tf.keras.Model(inputs, x, name=name)


def create_visuals(vis, x, y, y_pred):
    """
        Load visuals to visdom.

        Args:
            (utils.Visdom)  vis:    Visdom instance to use
            (np.ndarray)    x:      input
            (np.ndarray)    y:      ground-truth
            (np.ndarray)    y_pred: model prediction
    """
    # input
    source = np.empty((x.shape[0], 2*x.shape[1], 3), dtype=x.dtype)
    source[:, :x.shape[1]] = x[:, :, :3]
    source[:, x.shape[1]:] = x[:, :, 3:]
    vis.visdom.image(source.astype('uint8').transpose(2, 0, 1), win='source', opts={'title': 'source'})
    
    # ground-truth
    y = y*255.
    vis.visdom.image(y.astype('uint8').transpose(2, 0, 1), win='target', opts={'title': 'target'})

    # model prediction
    y_pred = y_pred*255.
    vis.visdom.image(y_pred.astype('uint8').transpose(2, 0, 1), win='prediction', opts={'title': 'prediction'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--clip_norm', type=float, default=100.0, help='threshold for gradient clipping (by norm)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to run')
    parser.add_argument('--first_epoch_decay', type=int, default=50, help='begin learing rate decay')
    parser.add_argument('--gpu', type=str, action='store', default='-1', help='GPU to use (e.g. \'0\' or \'1\'), default \'-1\' for no GPU')
    parser.add_argument('--L1', type=float, default=1., help='L1 loss factor')
    parser.add_argument('--L2', type=float, default=1., help='L2 loss factor')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.2, help='factor for learning rate decay')
    parser.add_argument('--lr_decay_freq', type=int, default=20, help='frequence of learning rate decay')
    parser.add_argument('--name', type=str, required=True, help='name of the model')
    parser.add_argument('--nb_samples', type=int, default=409204, help='number of samples to use')
    parser.add_argument('--out_path', type=str, required=True, help='path to the optical flows images')
    parser.add_argument('--in_path', type=str, required=True, help='path to the RGB images')
    parser.add_argument('--save_freq', type=int, default=10, help='frequence to save the model')
    parser.add_argument('--val_split', type=float, default=0.1, help='validation split')
    parser.add_argument('--visdom_env', type=str, required=True, help='visdom environment name')
    parser.add_argument('--visdom_port', type=int, action='store', default=8097, help='port for visdom to use')
    args = parser.parse_args()

    # construct visdom
    vis = utils.Visdom(args.visdom_env, args.visdom_port)
    assert vis.check_connection(), 'Visdom cannot connect'
    
    # display args
    d = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for p in [print, vis.log]:
        p('Program started at ' + d)
        p('Tensorflow version: ' + tf.__version__)
        p('-------------  OPTIONS  -------------')
        for k, v in vars(args).items():
            p('{:<20}: {}'.format(k, v))
        p('-------------------------------------')
    
    # define GPU(s) to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # build the model
    model = OpticalFlow(args.name)
    model.build(input_shape=(None, 256, 256, 6))

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, amsgrad=True, clipnorm=args.clip_norm)

    # loss function
    def loss_fn(y_true, y_pred, L1=args.L1, L2=args.L2):
        """ L1 + L2 + SSIM loss
        Args:
            (tf.Tensor) y_true: ground-truth
            (tf.Tensor) y_pred: model prediction
            (float)     L1:     factor for L1 loss
            (float)     L2:     factor for L2 loss
        
        Note: one should use SSIM with very small factor compared to L1/L2
        """
        L1_loss = L1*tf.keras.losses.mean_absolute_error(y_true, y_pred)
        L2_loss = L2*tf.keras.losses.mean_squared_error(y_true, y_pred)
        return L1_loss + L2_loss

    # Mean Squarred Error <=> End-Point Error (optical flow error function)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mean_squared_error', 'mean_absolute_error'])

    # EPIC-KITCHENS normalization values
    mean = tf.Variable([0.406, 0.456, 0.485, 0.406, 0.456, 0.485])  # repetition due to 2 RGB stacked
    std = tf.Variable([0.225, 0.224, 0.229, 0.225, 0.224, 0.229])  # repetition due to 2 RGB stacked
    
    # dataset
    start = time.time()
    data = CustomDataLoader(args.nb_samples, args.batch_size, args.val_split, args.in_path, args.out_path)
    end = time.time()
    for p in [print, vis.log]:
        p('\nTook {} minutes to load {} samples.'.format(round((end-start)/60), args.nb_samples))

    history = {'train': {}, 'val': {}}
    for epoch in range(args.epochs):
        epoch_history = {'train': {}, 'val': {}}
        for batch_idx, (batch, train) in enumerate(data):
            print('\rEpoch {}/{} - batch {}/{}      '.format(epoch + 1, args.epochs, batch_idx + 1, data.number_of_batches()), end='')
            x, y = batch
            vis_x, vis_y = x[0], y[0]
            x = (x/255. - mean)/std

            if train:
                results = model.train_on_batch(x, y, return_dict=True)
                split = 'train'
            else:
                results = model.test_on_batch(x, y, return_dict=True)
                split = 'val'
            for metric in results.keys():
                if metric not in epoch_history[split].keys():
                    epoch_history[split][metric] = [results[metric]]
                else:
                    epoch_history[split][metric].append(results[metric])
        """
            epoch_history = {
                train: {loss1: [..], loss2: [..], ..., metric1:[..]_, metric2: [..], ...},
                val: {loss1: [..], loss2: [..], ..., metric1:[..]_, metric2: [..], ...}
            }
        """

        # update history
        for metric in model.metrics_names:
            train_metric = sum(epoch_history['train'][metric])/len(epoch_history['train'][metric])
            val_metric = sum(epoch_history['val'][metric])/len(epoch_history['val'][metric])
            if metric not in history['train'].keys():
                history['train'][metric] = [train_metric]
                history['val'][metric] = [val_metric]
            else:
                history['train'][metric].append(train_metric)
                history['val'][metric].append(val_metric)
            print(' {}'.format(metric) + '{' + 'train: {}, val: {}'.format(train_metric, val_metric) + '}', end='')
        print('\n')  # go next line

        # render plots and visuals
        vis.plots(history)
        y_pred = model(x)
        create_visuals(vis, vis_x, vis_y, y_pred[0].numpy())

        # model checkpoint
        if (epoch + 1) % args.save_freq == 0:
            for p in [print, vis.log]:
                p('Saving model (epoch {})'.format(epoch + 1))
            model.save('models/{}/{}_{}.tf'.format(model.name, model.name, epoch + 1))
        
        # apply learning rate decay
        if ((epoch + 1) % args.lr_decay_freq == 0 and epoch + 1 > args.first_epoch_decay) or (epoch + 1 == args.first_epoch_decay):
            old_lr = optimizer.learning_rate.numpy()
            tf.keras.backend.set_value(optimizer.learning_rate, optimizer.learning_rate*args.lr_decay_factor)
            new_lr = optimizer.learning_rate.numpy()
            for p in [print, vis.log]:
                p('lr: {} -> {}'.format(old_lr, new_lr))
        
        # save environment
        vis.visdom.save([vis.env])

    # model last checkpoint
    for p in [print, vis.log]:
        p('Saving final model.')
    model.save('models/{}/{}_last.tf'.format(model.name, model.name))
    d = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for p in [print, vis.log]:
        p('Program finished at ' + d)
    
    # save environment
    vis.visdom.save([vis.env])
