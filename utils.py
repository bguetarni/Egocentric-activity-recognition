import tensorflow as tf
import visdom
import numpy as np
import pandas
import psutil
import random


class Visdom():
    def __init__(self, env, port):
        """
        Class to visualize plots and results (also log information)
        Args:
            (str) env: name of the environment; it's a good practice to define it as the model name
        """
        self.env = env
        self.port = port
        self.visdom = visdom.Visdom(env=self.env, port=self.port)
        self.log_win = 'logs'
        self.visdom.text('', win=self.log_win, opts={'title': 'Logs'})
    
    def plots(self, history):
        """
        Plot the curves (loss, accuracy...)
        Args:
            (dict) history: {
                                'train': {'loss1': [..], 'loss2': [..], ..., 'metric1': [..], 'metric2': [..], ...},
                                'val': {'loss1': [..], 'loss2': [..], ..., 'metric1': [..], 'metric2': [..], ...}
                            }
        """
        legends = ['train', 'val']
        X, Y = [], []
        for metric in history['train'].keys():
            X = [[i for i in range(len(history['train'][metric]))]]*2
            Y = [history['train'][metric], history['val'][metric]]
            X = np.array(X).transpose()
            Y = np.array(Y).transpose()
            self.visdom.line(Y, X, win=metric, opts={'xlabel': 'epoch', 'ylabel': metric, 'legend': legends})

    def log(self, log):
        """
            Log info/message
            Args:
                (str) log: text to log into visdom
        """
        self.visdom.text(log, win=self.log_win, append=True)
    
    def check_connection(self):
        return self.visdom.check_connection()


def get_TimeDistributed_MobilenetV2(name):
    """ Construct a MobileNetV2 backbone """
    
    # load MboileNetV2 pre-trained on ImageNet
    mobilenet = tf.keras.applications.MobileNetV2(input_shape=(256, 256, 3), include_top=False)
    time_distributed_mobilenet = tf.keras.layers.TimeDistributed(mobilenet)
    time_distributed_mobilenet.trainable = False
    return time_distributed_mobilenet


def reshape_output_labels(y, verbs_categories, nouns_categories=None):
    """
    Reshape the outputs labels when using categories
    Args:
        (np.ndarray/list)   y:                  labels ->   numpy.ndarray (if verbs only)
                                                            list of numpy.ndarray (if verbs and nouns) -> [verbs, nouns]
        (dict)              verbs_categories:   categories and associated verbs
        (dict/None)         nouns_categories:   categories and associated nouns ->  dict (if verbs and nouns)
                                                                                    None (if verbs only)
    """

    # classes ID
    with open('data/EPIC_100_verb_classes.csv', 'r') as f:
        verbs_classes = pandas.read_csv(f)
    with open('data/EPIC_100_noun_classes.csv', 'r') as f:
        noun_classes = pandas.read_csv(f)
    
    if isinstance(y, list):
        y = np.array(y)
    
    if nouns_categories is None:
        outputs = np.zeros(shape=(y.shape[0], len(verbs_categories)))
        for sample_nb, out in enumerate(y):
            # current class ID
            # id of element with max probability
            idx = np.argmax(out)
            
            # class key
            # key of class associated with element with max probability
            class_key = verbs_classes.at[idx, 'key']
            
            # look for category containing current class and set to one
            for i, category_classes_key in enumerate(verbs_categories.values()):
                if class_key in category_classes_key:
                    outputs[sample_nb][i] = 1
                    break
    else:  # same thing but for 2 labels
        verbs_outputs = np.zeros(shape=(y[0].shape[0], len(verbs_categories)))
        nouns_outputs = np.zeros(shape=(y[1].shape[0], len(nouns_categories)))
        assert verbs_outputs.shape[0] == nouns_outputs.shape[0], 'verbs and nouns must have same 1st dimension (number of samples)'

        for sample_nb, (verb_out, noun_out) in enumerate(zip(y[0], y[1])):
            verb_idx = np.argmax(verb_out)
            noun_idx = np.argmax(noun_out)

            verb_class_key = verbs_classes.at[verb_idx, 'key']
            noun_class_key = noun_classes.at[noun_idx, 'key']

            for i, category_classes_key in enumerate(verbs_categories.values()):
                if verb_class_key in category_classes_key:
                    verbs_outputs[sample_nb][i] = 1
                    break
            for i, category_classes_key in enumerate(nouns_categories.values()):
                if noun_class_key in category_classes_key:
                    nouns_outputs[sample_nb][i] = 1
                    break
        outputs = [verbs_outputs, nouns_outputs]
    return outputs


def get_epic55_train_videos():
    """ Return the training set videos of EPIC-KITCHENS-55 that are present in EPIC-KITCHENS-100 """

    p = pandas.read_csv('data/epic_55_splits.csv')
    videos = []
    for _, line in p.iterrows():
        if line['split'] == 'train':
            videos.append(line['video_id'])
    return videos


def create_groups(memory_percent, seq_length, version, split=0.2):
    """
    Create groups of videos for training/validation, each group take {memory_percent}% of the RAM.
    Use videos dependantly of which version of the dataset specified.
    Args:
        (float) memory_percent: percentage of RAM to use in float type (e.g. 35% -> 0.35)
        (int)   seq_length:     number of images to use in the time-serie sequence
        (int)   version:        version of the dataset (55 or 100)
        (float) split:          split factor for validation set
    """

    shapes = {}
    # extract videos shapes from CSV training file
    p = pandas.read_csv('data/EPIC_100_train.csv')
    for _, line in p.iterrows():
        v = line['video_id']
        if v not in shapes.keys():
            shapes[v] = 0
        shapes[v] += 1
    p = pandas.read_csv('data/EPIC_100_validation.csv')
    for _, line in p.iterrows():
        v = line['video_id']
        if v not in shapes.keys():
            shapes[v] = 0
        shapes[v] += 1
    
    # define memory size for 1 action (uint8)
    if seq_length == 4:
        one_action_footprint = 786608
    else:
        one_action_footprint = 3932336
    
    total_memory = psutil.virtual_memory().total - 1024**3  # (RAM available - 1Go; precaution)
    memory_usage = memory_percent*total_memory
    nb_actions = round(memory_usage/one_action_footprint)-1

    train_groups, val_groups = [], []
    if version == 100:
        # extract splits
        splits = {'train': [], 'val': []}
        p = pandas.read_csv('data/epic_100_splits.csv')
        for _, line in p.iterrows():
            if line['ar_train']:
                splits['train'].append(line['video_id'])
            elif line['ar_val']:
                splits['val'].append(line['video_id'])
        random.shuffle(splits['train'])
        random.shuffle(splits['val'])

        # computes groups accordingly to the RAM specified
        while len(splits['train']) > 0:
            # if current group small enough, add current video to it
            if len(train_groups) > 0 and train_groups[-1][1] < nb_actions:
                v = splits['train'].pop()
                train_groups[-1][0].append(v)
                train_groups[-1][1] += shapes[v]
                train_groups[-1][2] = True
            else:  # otherwise create a new group
                v = splits['train'].pop()
                train_groups.append([[v], shapes[v], True])
        
        # same than before but for validation groups
        while len(splits['val']) > 0:
            if len(val_groups) > 0 and val_groups[-1][1] < nb_actions:
                v = splits['val'].pop()
                val_groups[-1][0].append(v)
                val_groups[-1][1] += shapes[v]
                val_groups[-1][2] = False
            else:
                v = splits['val'].pop()
                val_groups.append([[v], shapes[v], False])

        return train_groups + val_groups
    elif version == 55:
        videos = get_epic55_train_videos()
        random.shuffle(videos)
        
        total_samples = 0
        for v in videos:
            total_samples += shapes[v]
        split_sample = round((1 - split)*total_samples)
        
        # training samples
        nb_train_actions = 0
        while nb_train_actions < split_sample:
            if len(train_groups) > 0 and train_groups[-1][1] < nb_actions:
                v = videos.pop()
                train_groups[-1][0].append(v)
                train_groups[-1][1] += shapes[v]
                train_groups[-1][2] = True
            else:
                v = videos.pop()
                train_groups.append([[v], shapes[v], True])
            nb_train_actions += shapes[v]

        # validation samples
        while len(videos) > 0:
            if len(val_groups) > 0 and val_groups[-1][1] < nb_actions:
                v = videos.pop()
                val_groups[-1][0].append(v)
                val_groups[-1][1] += shapes[v]
                val_groups[-1][2] = False
            else:
                v = videos.pop()
                val_groups.append([[v], shapes[v], False])

        return train_groups + val_groups
    else:
        raise ValueError('ERROR: dataset version specified is incorrect. Specified etiher 55 or 100')
