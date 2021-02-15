import tensorflow as tf
import numpy as np
import argparse
import math
import os
import yaml
import utils
import pandas
from matplotlib import pyplot as plt
import seaborn
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, action='store', default=16, help='batche size')
    parser.add_argument('--data', type=str, action='store', required=True, help='path to the folder with test data')
    parser.add_argument('--gpu', type=str, action='store', default='-1', help='which GPU to use (e.g. \'0\'), default \'-1\' for no GPU')
    parser.add_argument('--models', type=str, action='store', required=True, help='path to the folder that contain models to test')
    args = parser.parse_args()
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
    
    # display args
    print('-------------  OTPIONS  -------------')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print('-------------------------------------')

    # define GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # deactivate tensorflow graph optimization to avoid 'detected edge(s) creating cycle(s)' errors
    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

    # categories
    with open('data/EPIC_100_verbs_categories.yaml', 'r') as fv, open('data/EPIC_100_nouns_categories.yaml', 'r') as fn:
        verbs_categories = yaml.load(fv, Loader=yaml.FullLoader)
        nouns_categories = yaml.load(fn, Loader=yaml.FullLoader)
    
    # load verb classes
    verbs_classes = {}
    verb_labels = pandas.read_csv('data/EPIC_100_verb_classes.csv')
    for _, line in verb_labels.iterrows():
        verbs_classes.update({line.at['key']: int(line.at['id'])})

    # load noun classes
    nouns_classes = {}
    noun_labels = pandas.read_csv('data/EPIC_100_noun_classes.csv')
    for _, line in noun_labels.iterrows():
        nouns_classes.update({line.at['key']: int(line.at['id'])})
    
    # load verb categories
    with open('data/EPIC_100_verbs_categories.yaml', 'r') as f:
        categories = yaml.load(f)
    verbs_categories_classes = {}
    for i, k in enumerate(categories.keys()):
        verbs_categories_classes[k] = i

    # load noun categories
    with open('data/EPIC_100_nouns_categories.yaml', 'r') as f:
        categories = yaml.load(f)
    nouns_categories_classes = {}
    for i, k in enumerate(categories.keys()):
        nouns_categories_classes[k] = i

    # we computes the dataset size beforehand to reserves the space in order to avoid doing big copies during its construction
    videos = os.listdir(args.data)
    size = 0
    p = pandas.read_csv('data/EPIC_100_validation.csv')
    for _, line in p.iterrows():
        if line['video_id'] + '.npz' in videos:
            size += 1

    X = np.empty(shape=(size, 20, 256, 256, 3), dtype='uint8')
    Y = [np.empty(shape=(size, 97), dtype='float32'), np.empty(shape=(size, 300), dtype='float32')]
    n = 0
    for i, v in enumerate(videos):
        print('\rLoading data {}/{}'.format(i+1, len(videos)), end='')
        data = np.load(args.data + v, allow_pickle=True)
        x, y = data['x'], [data['y_verb'], data['y_noun']]
        X[n:n+x.shape[0]] = x
        Y[0][n:n+x.shape[0]] = y[0]
        Y[1][n:n+x.shape[0]] = y[1]
        n += x.shape[0]
    print('\ndata loaded: {} samples'.format(X.shape[0]))
    nb_batches = math.ceil(X.shape[0]/args.batch_size)  # number of batchs
    
    for model_chkpt in os.listdir(args.models):
        # load saved model
        model = tf.keras.models.load_model(model_chkpt, compile=False)
        print(model.name)
        
        if isinstance(model.layers[-1].output_shape, list):
            verb_only = False
            if model.layers[-1].output_shape[0][1] == 13:
                use_categories = True
            else:
                use_categories = False
        else:
            verb_only = True
            if model.layers[-1].output_shape[1] == 13:
                use_categories = True
            else:
                use_categories = False

        y_true_verb, y_pred_verb, y_true_noun, y_pred_noun = None, None, None, None
        for step in range(nb_batches):  # compute predictions for video
            print('\r{} - batch {}/{}'.format(model.name, step+1, nb_batches), end='')
            if step < (nb_batches - 1):
                idx = range(step*args.batch_size, (step+1)*args.batch_size)
            else:
                idx = range(step*args.batch_size, X.shape[0])
            
            x_step = X[idx].astype('float32')
            if verb_only:
                y_step = Y[0][idx]
            else:
                y_step = [Y[0][idx], Y[1][idx]]
            
            y = model(x_step)

            if verb_only:
                if use_categories:  # change outputs if categories have been used
                    y_step = utils.reshape_output_labels(y_step, verbs_categories)
                
                if y_true_verb is None:  # first batch
                    y_true_verb = y_step
                    y_pred_verb = y.numpy()
                else:
                    y_true_verb = np.concatenate((y_true_verb, y_step), axis=0)  # verb ground-truth
                    y_pred_verb = np.concatenate((y_pred_verb, y.numpy()), axis=0)  # verb prediction
            else:
                if use_categories:
                    y_step = utils.reshape_output_labels(y_step, verbs_categories, nouns_categories)
                
                if y_true_verb is None:  # first batch
                    y_true_verb = y_step[0]
                    y_pred_verb = y[0].numpy()
                    y_true_noun = y_step[1]
                    y_pred_noun = y[1].numpy()
                else:
                    y_true_verb = np.concatenate((y_true_verb, y_step[0]), axis=0)  # verb ground-truth
                    y_pred_verb = np.concatenate((y_pred_verb, y[0].numpy()), axis=0)  # verb prediction
                    y_true_noun = np.concatenate((y_true_noun, y_step[1]), axis=0)  # noun ground-truth
                    y_pred_noun = np.concatenate((y_pred_noun, y[1].numpy()), axis=0)  # noun prediction
        
        # get the labels (e.g. class 1 -> 0, class 2 -> 1, ...)
        if not use_categories:
            verb_labels = list(verbs_classes.values())
            noun_labels = list(nouns_classes.values())
        else:
            verb_labels = list(verbs_categories_classes.values())
            noun_labels = list(nouns_categories_classes.values())

        # compute and display the metrics
        print('\nCompute metrics')
        y_true_verb_labels = np.argmax(y_true_verb, axis=-1)
        y_pred_verb_labels = np.argmax(y_pred_verb, axis=-1)
        metrics = {
            'verb_top_1_accuracy': [accuracy_score(y_true_verb_labels, y_pred_verb_labels)],
            'verb_top_3_accuracy': [top_k_accuracy_score(y_true_verb_labels, y_pred_verb, k=3, labels=verb_labels)],
            'verb_top_5_accuracy': [top_k_accuracy_score(y_true_verb_labels, y_pred_verb, k=5, labels=verb_labels)],
            'verb_confusion_matrix': [confusion_matrix(y_true_verb_labels, y_pred_verb_labels, labels=verb_labels)]
        }

        if not verb_only:
            # noun accuracy
            y_true_noun_labels = np.argmax(y_true_noun, axis=-1)
            y_pred_noun_labels = np.argmax(y_pred_noun, axis=-1)
            metrics['noun_top_1_accuracy'] = accuracy_score(y_true_noun_labels, y_pred_noun_labels)
            metrics['noun_top_3_accuracy'] = top_k_accuracy_score(y_true_noun_labels, y_pred_noun, k=3, labels=noun_labels)
            metrics['noun_top_5_accuracy'] = top_k_accuracy_score(y_true_noun_labels, y_pred_noun, k=5, labels=noun_labels)
            metrics['noun_confusion_matrix'] = confusion_matrix(y_true_noun_labels, y_pred_noun_labels, labels=noun_labels)
            
            # action accuracy (ugly but works !)
            y_true_combined = np.empty((y_true_verb.shape[0], y_true_verb.shape[1]*y_true_noun.shape[1]), dtype=y_true_verb.dtype)
            y_pred_combined = np.empty((y_pred_verb.shape[0], y_pred_verb.shape[1]*y_pred_noun.shape[1]), dtype=y_pred_verb.dtype)
            for i in range(y_true_combined.shape[0]):
                for j in range(y_true_verb.shape[1]):
                    y_true_combined[i, y_true_verb.shape[1]*j:y_true_verb.shape[1]*(j+1)] = y_true_verb[i]*y_true_noun[i][j]
                    y_pred_combined[i, y_pred_verb.shape[1]*j:y_pred_verb.shape[1]*(j+1)] = y_pred_verb[i]*y_pred_noun[i][j]
            y_true_combined_labels = np.argmax(y_true_combined, axis=-1)
            combined_labels = [i for i in range(y_true_verb.shape[1]*y_true_noun.shape[1])]
            metrics['action_top_1_accuracy'] = accuracy_score(y_true_combined_labels, np.argmax(y_pred_combined, axis=-1))
            metrics['action_top_3_accuracy'] = top_k_accuracy_score(y_true_combined_labels, y_pred_combined, k=3, labels=combined_labels)
            metrics['action_top_5_accuracy'] = top_k_accuracy_score(y_true_combined_labels, y_pred_combined, k=5, labels=combined_labels)
        
        # display metrics
        for metric_name, metric_value in metrics.items():
            if 'confusion_matrix' in metric_name:  # plot and save the confusion matrices
                if use_categories:
                    if 'verb' in metric_name:
                        labels = verbs_categories_classes.keys()
                    else:
                        labels = nouns_categories_classes.keys()
                else:
                    if 'verb' in metric_name:
                        labels = verbs_classes.keys()
                    else:
                        labels = nouns_classes.keys()
                plt.subplots(figsize=(15, 15))
                heatmap = seaborn.heatmap(np.round(metric_value, 2), annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                heatmap_name = '{}_{}.png'.format(model.name, metric_name)
                heatmap.get_figure().savefig(fname=heatmap_name)
                print('     confusion matrix saved in ' + heatmap_name)
            else:  # display metric
                print('     {}: {}'.format(metric_name, [round(i, 4) for i in metric_value]))
