import tensorflow as tf
import gc
import numpy as np
import math
import datetime
import time
import logging
import os
import argparse
import yaml
import utils
import model
import Attention


def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        if isinstance(y_pred, list):
            verb_loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y[0], y_pred[0])
            noun_loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y[1], y_pred[1])
            loss = verb_loss + noun_loss
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y, y_pred)
    
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    for metric in metrics:
        if isinstance(y_pred, list):
            metric.update_state(y[0], y_pred[0])
            metric.update_state(y[1], y_pred[1])
        else:
            metric.update_state(y, y_pred)
    return loss


def test_step(x, y):
    y_pred = model(x, training=False)
    if isinstance(y_pred, list):
        verb_loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y[0], y_pred[0])
        noun_loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y[1], y_pred[1])
        loss = verb_loss + noun_loss
    else:
        loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y, y_pred)
    
    for metric in metrics:
        if isinstance(y_pred, list):
            metric.update_state(y[0], y_pred[0])
            metric.update_state(y[1], y_pred[1])
        else:
            metric.update_state(y, y_pred)
    return loss


# train step defined as Tensorflow function for speed enhancement
@tf.function
def distributed_train_step(x, y):
    """ training step on multi-GPU """
    per_replica_losses = strategy.run(train_step, args=(x, y))
    loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=0)  # average the batches loss across all replicas (GPUs)
    return loss


# test step defined as Tensorflow function for speed enhancement
@tf.function
def distributed_test_step(x, y):
    """ test step on multi-GPU """
    per_replica_losses = strategy.run(test_step, args=(x, y))
    loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=0)  # average the batches loss across all replicas (GPUs)
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, action='store', default=16, help='(global) batch size')
    parser.add_argument('--clip_norm', type=float, default=100.0, help='threshold for gradient clipping (by norm)')
    parser.add_argument('--data', type=str, action='store', required=True, help='path to the training set folder')
    parser.add_argument('--dataset_version', type=int, action='store', default=100, help='version of EPIC-KITCHENS (55/100)')
    parser.add_argument('--epochs', type=int, action='store', default=100, help='number of epochs to train the model')
    parser.add_argument('--epochs_per_group', type=int, action='store', default=1, help='number of epochs each group run before loading next group')
    parser.add_argument('--first_epoch_decay', type=int, action='store', default=50, help='epoch to begin learning rate decay')
    parser.add_argument('--gpu', type=str, action='store', default='-1', help='GPU to use (e.g. \'1\' or \'0,1\'), default \'-1\' for no GPU')
    parser.add_argument('--lr', type=float, action='store', default=0.01, help='starting learning rate')
    parser.add_argument('--lr_decay_factor', type=float, action='store', default=0.1, help='factor by which decreasing learning rate')
    parser.add_argument('--lr_decay_freq', type=int, action='store', default=20, help='frequency of decreasing learning rate')
    parser.add_argument('--memory_ratio', type=float, action='store', default=0.4, help='amount of RAM to use')
    parser.add_argument('--name', type=str, action='store', required=True, help='name to give to the model')
    parser.add_argument('--nb_frames', type=int, action='store', default=20, help='sequence length')
    parser.add_argument('--optimizer', type=str, action='store', default='adam', help='name of the optimizer: adam | sgd | rmsprop')
    parser.add_argument('--save_freq', type=int, action='store', default=10, help='frequency of saving the model')
    parser.add_argument('--type', type=str, action='store', required=True, help='time serie layer type: ConvLSTM | LSTM | ConvNet1D | AttentionTemporalCNN')
    parser.add_argument('--use_categories', type=str, action='store', default="False", help='wether to use categories rather than all classes')
    parser.add_argument('--verb_only', type=str, action='store', default="False", help='wether to use only verbs')
    parser.add_argument('--visdom_env', type=str, action='store', required=True, help='visdom environment name')
    parser.add_argument('--visdom_port', type=int, action='store', default=8097, help='port for visdom to use')
    args = parser.parse_args()
    args.use_categories = eval(args.use_categories)
    args.verb_only = eval(args.verb_only)
    """
    data folder must be like:
        data\
            PXX_XX.npz
            PXX_XX.npz
            PXX_XX.npz
            .
            .
            .
    """

    # construct visdom
    vis = utils.Visdom(args.visdom_env, args.visdom_port)
    if not vis.check_connection():
        print('Visdom cannot connect. Start a visdom session before starting training (see README)')
        exit()

    # logger for debugging
    logger = logging.getLogger(__name__)
    log_name = datetime.datetime.now().strftime("%d_%m-%H:%M")
    f = logging.FileHandler('logs/' + log_name + '.log', mode='w')
    f.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(f)
    logger.setLevel(logging.DEBUG)
    
    # display args
    d = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for p in [print, logger.info, vis.log]:
        p('Program started at ' + d)
        p('Tensorflow version: ' + tf.__version__)
        p('-------------  OPTIONS  -------------')
        for k, v in vars(args).items():
            p('{:<20}: {}'.format(k, v))
        p('-------------------------------------')

    # number of classes
    if args.use_categories:
        nb_verbs = 13
        nb_nouns = 21
    else:
        nb_verbs = 97
        nb_nouns = 300
    
    # path to the traning set
    DATA_PATH = args.data
    DATA_PATH = (DATA_PATH + '/') if DATA_PATH[-1] != '/' else DATA_PATH
    
    # define GPU(s) to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # send the normalization values to GPU(s); [R, G, B]
    mean = tf.constant([0.485, 0.456, 0.406], dtype='float32')
    std = tf.constant([0.229, 0.224, 0.225], dtype='float32')

    # deactivate tensorflow graph optimization to avoid 'detected edge(s) creating cycle(s)' errors
    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    
    # define strategy for multiGPU
    strategy = tf.distribute.MirroredStrategy()

    # create the model
    with strategy.scope():
        if args.type in ['ConvLSTM', 'LSTM', 'ConvNet1D']:
            model = model.Model(_name=args.name, _type=args.type, _verb_only=args.verb_only, out_verbs=nb_verbs, out_nouns=nb_nouns)
        elif args.type == 'AttentionTemporalCNN':
            model = Attention.AttentionTemporalCNN(name=args.name, verb_only=args.verb_only, out_verbs=nb_verbs, out_nouns=nb_nouns)
        else:
            raise NameError('Model type must be one of: ConvLSTM | LSTM | ConvNet1D | AttentionTemporalCNN')

    # build the model and print its summary
    model.build(input_shape=(None, None, 256, 256, 3))
    model.summary(line_length=150)

    # load categories
    if args.use_categories:
        with open('data/EPIC_100_verbs_categories.yaml', 'r') as fv, open('data/EPIC_100_nouns_categories.yaml', 'r') as fn:
            verbs_categories = yaml.load(fv, Loader=yaml.FullLoader)
            nouns_categories = yaml.load(fn, Loader=yaml.FullLoader)
    
    # utilities
    with strategy.scope():
        if args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, clipnorm=args.clip_norm)
        elif args.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr, momentum=0.9, clipnorm=args.clip_norm)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=args.clip_norm)
        metrics = [tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(top_k=5)]

    history = {'train': {}, 'val': {}}
    groups = utils.create_groups(memory_percent=args.memory_ratio, seq_length=args.nb_frames, version=args.dataset_version)  # create groups of videos to train
    for epoch in range(args.epochs//args.epochs_per_group):
        epoch_history = {'train': [], 'val': []}

        # loop over the groups
        for videos, size, train in groups:
            X = np.empty(shape=(size, args.nb_frames, 256, 256, 3), dtype='uint8')
            Y = None

            i = 0
            start = time.time()
            for v in videos:  # load training/validation data
                print('\r   Loading actions: {}/{}     '.format(i, X.shape[0]), end='')
                data = np.load(DATA_PATH + v + '.npz', allow_pickle=True)
                if model.verb_only:
                    x, y = data['x'], data['y_verb']
                else:
                    x, y = data['x'], [data['y_verb'], data['y_noun']]

                if isinstance(y, list):  # check there are as many labels as images
                    b = (x.shape[0] == y[0].shape[0] and x.shape[0] == y[1].shape[0])
                else:
                    b = (x.shape[0] == y.shape[0])
                if not b:
                    for p in [logger.warning, vis.log]:
                        p('X and Y are not the same shape -> skipping video : {}'.format(v))
                    continue

                if Y is None:
                    Y = y
                else:
                    if args.verb_only:
                        Y = np.concatenate((Y, y), axis=0)
                    else:  # Y = [verbs, nouns]
                        Y[0] = np.concatenate((Y[0], y[0]), axis=0)
                        Y[1] = np.concatenate((Y[1], y[1]), axis=0)

                X[i:i+x.shape[0]] = x
                i += x.shape[0]
            del x, y
            end = time.time()
            for p in [logger.info, vis.log]:
                p('Loading took {} mins'.format(round((end-start)/60)))

            if (X is not None) and (len(X.shape) == 5) and (X.shape[0] > 1):
                if args.use_categories:
                    if args.verb_only:
                        Y = utils.reshape_output_labels(Y, verbs_categories)
                    else:
                        Y = utils.reshape_output_labels(Y, verbs_categories, nouns_categories)
                
                # number of batchs
                nb_batchs = math.ceil(X.shape[0]/args.batch_size)

                # ====================================== TRAINING / VALIDATION ======================================
                for sub_epoch in range(args.epochs_per_group):
                    epoch_history['train'].append({})
                    epoch_history['val'].append({})

                    for step in range(nb_batchs):  # loop over batchs
                        print('\r   Epoch {}/{} - batch {}/{}      '.format(epoch*args.epochs_per_group + (sub_epoch + 1), args.epochs, step+1, nb_batchs), end='')
                        if step < (nb_batchs - 1):  # not last batch
                            indices = range(step*args.batch_size, (step+1)*args.batch_size)
                        else:  # last batch => size may be less than batch size !
                            indices = range(step*args.batch_size, X.shape[0])
                        
                        # send the inputs/outputs to the GPU
                        x_step = X[indices].astype('float32')
                        if args.verb_only:
                            y_step = Y[indices].astype('float32')
                        else:
                            y_step = [Y[0][indices].astype('float32'), Y[1][indices].astype('float32')]

                        # input normalization
                        x_step = (x_step/255. - mean)/std
                        
                        # 1 train/test step
                        if train:
                            loss = distributed_train_step(x_step, y_step)
                            split = 'train'
                        else:
                            loss = distributed_test_step(x_step, y_step)
                            split = 'val'
                        
                        results = {'loss': loss}
                        for metric in metrics:
                            results.update({metric.name: metric.result().numpy()})
                            metric.reset_states()
                        
                        for metric_name, metric_results in results.items():
                            if metric_name not in epoch_history[split][sub_epoch].keys():
                                epoch_history[split][sub_epoch][metric_name] = [metric_results]
                            else:
                                epoch_history[split][sub_epoch][metric_name].append(metric_results)
                # ===================================================================================================
            # free memory before loading next samples
            del X, Y, x_step, y_step
            tf.keras.backend.clear_session()
            gc.collect()
        
        d = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for p in [print, logger.info, vis.log]:
            p('Finished epochs {} --> {}.'.format(args.epochs_per_group*epoch + 1, args.epochs_per_group*(epoch + 1)))
            p(d)

        # update history
        for sub_epoch in range(args.epochs_per_group):
            for metric in results.keys():
                train_metric = sum(epoch_history['train'][sub_epoch][metric])/len(epoch_history['train'][sub_epoch][metric])
                val_metric = sum(epoch_history['val'][sub_epoch][metric])/len(epoch_history['val'][sub_epoch][metric])
                if metric not in history['train'].keys():
                    history['train'][metric] = [train_metric]
                    history['val'][metric] = [val_metric]
                else:
                    history['train'][metric].append(train_metric)
                    history['val'][metric].append(val_metric)
                print(' {}'.format(metric) + '{' + 'train: {}, val: {}'.format(train_metric, val_metric) + '}', end='')
            print('\n')  # go next line
        vis.plots(history)  # render epoch results

        # model checkpoint
        if args.epochs_per_group*(epoch+1) % args.save_freq == 0:
            for p in [print, logger.info, vis.log]:
                p('Saving model (epoch {})'.format(args.epochs_per_group*(epoch+1)))
            
            if model.name not in os.listdir('models/'):
                os.mkdir('models/{}'.format(model.name))
            model.save('models/{}/{}_{}.tf'.format(model.name, model.name, args.epochs_per_group*(epoch+1)))
        
        # apply learning rate decay
        if (args.epochs_per_group*(epoch+1) % args.lr_decay_freq == 0 and args.epochs_per_group*(epoch+1) > args.first_epoch_decay) or (args.epochs_per_group*(epoch+1) == args.first_epoch_decay):
            old_lr = optimizer.learning_rate.numpy()
            tf.keras.backend.set_value(optimizer.learning_rate, optimizer.learning_rate*args.lr_decay_factor)
            new_lr = optimizer.learning_rate.numpy()
            for p in [print, logger.info, vis.log]:
                p('lr: {} -> {}'.format(old_lr, new_lr))
        
        # save environment
        vis.visdom.save([vis.env])

    # model last checkpoint
    for p in [print, logger.info, vis.log]:
        p('Saving final model.')
    model.save('models/{}/{}_last.tf'.format(model.name, model.name))
    d = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for p in [print, logger.info, vis.log]:
        p('Program finished at ' + d)
