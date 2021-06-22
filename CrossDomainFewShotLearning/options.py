import numpy as np
import os
import glob
import torch
import argparse
import json

def parse_args(script):
    parser = argparse.ArgumentParser(
        description='few-shot script %s' % (script))
    parser.add_argument('--dataset', default='multi',
                        help='miniImagenet/cub/cars/places/plantae, specify multi for training with multiple domains')
    parser.add_argument('--testset', default='cub',
                        help='cub/cars/places/plantae, valid only when dataset=multi')
    # we use ResNet10 in the paper
    parser.add_argument('--model', default='ResNet10',
                        help='model: Conv{4|6} / ResNet{10|18|34}')
    parser.add_argument('--method', default='baseline',
                        help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/gnnnet')
    parser.add_argument('--train_n_way', default=5, type=int,
                        help='class num to classify for training')
    parser.add_argument('--test_n_way', default=5, type=int,
                        help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot', default=5, type=int,
                        help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=16, type=int,
                        help='number of query data in each class')
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training ')
    parser.add_argument('--name', default='tmp', type=str, help='')
    parser.add_argument('--filepath_to_arguments_json_file',
                        default=None, type=str, help='')
    parser.add_argument('--save_dir', default='./output', type=str, help='')
    parser.add_argument('--data_dir', default='./filelists', type=str, help='')

    if script == 'train':
        parser.add_argument('--num_classes', default=200, type=int,
                            help='total number of classes in softmax, only used in baseline')
        parser.add_argument('--save_freq', default=25,
                            type=int, help='Save frequency')
        parser.add_argument('--start_epoch', default=0,
                            type=int, help='Starting epoch')
        parser.add_argument('--stop_epoch', default=400,
                            type=int, help='Stopping epoch')
        parser.add_argument('--resume', default='', type=str,
                            help='continue from previous trained model with largest epoch')
        parser.add_argument('--resume_epoch', default=-1, type=int, help='')
        parser.add_argument('--warmup', default='gg3b0', type=str,
                            help='continue from baseline, neglected if resume is true')
        parser.add_argument('--n_model_candidates', default=2,
                            type=int, help='Number of model candidates')
        parser.add_argument('--temperature', default=0.05,
                            type=float, help='Temperature for meta-evolution')
        parser.add_argument('--use_ft', default='no', type=str,
                            help='if to use fixed ft layers')
    elif script == 'test':
        parser.add_argument('--split', default='novel', help='base/val/novel')
        parser.add_argument('--save_epoch', default=400, type=int,
                            help='load the model trained in x epoch, use the best model if x is -1')
    else:
        raise ValueError('Unknown script')

    args = parser.parse_args()

    if args.filepath_to_arguments_json_file is not None:
        args = extract_args_from_json(
            json_file_path=args.filepath_to_arguments_json_file, existing_args_dict=args)
    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print(arg_str)

    return args


class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(json_file_path, existing_args_dict=None):

    summary_filename = json_file_path
    with open(summary_filename) as f:
        arguments_dict = json.load(fp=f)

    for key, value in vars(existing_args_dict).items():
        if key not in arguments_dict:
            arguments_dict[key] = value

    arguments_dict = AttributeAccessibleDict(arguments_dict)

    return arguments_dict


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir, resume_epoch=-1):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0])
                       for x in filelist])
    max_epoch = np.max(epochs)
    epoch = max_epoch if resume_epoch == -1 else resume_epoch
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def load_warmup_state(filename, method, device):
    print('  load pre-trained model file: {}'.format(filename))
    warmup_resume_file = get_resume_file(filename)
    tmp = torch.load(warmup_resume_file, map_location=device)

    if tmp is not None:
        state = tmp['state']
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if 'relationnet' in method and "feature." in key:
                newkey = key.replace("feature.", "")
                state[newkey] = state.pop(key)
            elif method == 'gnnnet' and 'feature.' in key:
                newkey = key.replace("feature.", "")
                state[newkey] = state.pop(key)
            elif method == 'matchingnet' and 'feature.' in key and '.7.' not in key:
                newkey = key.replace("feature.", "")
                state[newkey] = state.pop(key)
            else:
                state.pop(key)
    else:
        raise ValueError(' No pre-trained encoder file found!')
    return state
