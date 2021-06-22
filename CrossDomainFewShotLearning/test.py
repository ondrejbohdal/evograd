import torch
import os
import h5py
import json

from methods import backbone
from methods.backbone import model_dict
from data.datamgr import SimpleDataManager
from options import parse_args, get_best_file, get_assigned_file

from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.gnnnet import GnnNet
from methods.relationnet import RelationNet
from methods.patched_relationnet import PatchedRelationNet
import data.feature_loader as feat_loader
import random
import numpy as np

from higher.patch import buffer_sync
from higher.patch import make_functional
from higher.utils import get_func_params

# extract and save image features


def save_features(model, data_loader, featurefile, device):
    f = h5py.File(featurefile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0
    for i, (x, y) in enumerate(data_loader):
        if (i % 10) == 0:
            print('    {:d}/{:d}'.format(i, len(data_loader)))
        x = x.to(device=device)
        feats = model(x)
        if all_feats is None:
            all_feats = f.create_dataset(
                'all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    f.close()


def save_features_patched(model, params, data_loader, featurefile, device):
    patched_model = make_functional(model)
    buffer_sync(model, patched_model)

    f = h5py.File(featurefile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0
    for i, (x, y) in enumerate(data_loader):
        if (i % 10) == 0:
            print('    {:d}/{:d}'.format(i, len(data_loader)))
        x = x.to(device=device)
        feats = patched_model(x, params=params)
        if all_feats is None:
            all_feats = f.create_dataset(
                'all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    f.close()


# evaluate using features

def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]])
                      for i in range(n_support+n_query)])
    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y)*100
    return acc


def feature_evaluation_patched(cl_data_file, model, n_way=5, n_support=5, n_query=15):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]])
                      for i in range(n_support+n_query)])
    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query

    scores = model.set_forward(z_all, model.feature_params, model.rm_params, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y)*100
    return acc


def update_json_experiment_log_dict(experiment_update_dict, params):
    json_experiment_log_file_name = os.path.join(
        'results', params.name) + '.json'
    with open(json_experiment_log_file_name, 'r') as f:
        summary_dict = json.load(fp=f)

    for key in experiment_update_dict:
        summary_dict[key].append(experiment_update_dict[key])

    with open(json_experiment_log_file_name, 'w') as f:
        json.dump(summary_dict, fp=f)


# --- main ---
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("use GPU", device)
    else:
        device = torch.device('cpu')
        print("use CPU")

    # parse argument
    params = parse_args('test')
    print('Testing! {} shots on {} dataset with {} epochs of {}({})'.format(
        params.n_shot, params.testset, params.save_epoch, params.name, params.method))
    remove_featurefile = True

    print('\nStage 1: saving features')
    # dataset
    print('  build dataset')
    if 'Conv' in params.model:
        image_size = 84
    else:
        image_size = 224
    split = params.split
    loadfile = os.path.join(params.data_dir, params.testset, split + '.json')
    datamgr = SimpleDataManager(image_size, batch_size=64)
    data_loader = datamgr.get_data_loader(loadfile, aug=False)

    print('  build feature encoder')
    # feature encoder
    checkpoint_dir = '%s/checkpoints/%s' % (params.save_dir, params.name)
    if params.save_epoch != -1:
        modelfile = get_assigned_file(checkpoint_dir, params.save_epoch)
    else:
        modelfile = get_best_file(checkpoint_dir)

    if params.method in ['relationnet', 'relationnet_softmax', 'relationnet_evo', 'relationnet_softmax_evo']:
        if params.model == 'Conv4':
            model = backbone.Conv4NP()
        elif params.model == 'Conv6':
            model = backbone.Conv6NP()
        else:
            model = model_dict[params.model](flatten=False)
    else:
        model = model_dict[params.model]()
    model = model.to(device=device)
    tmp = torch.load(modelfile, map_location=device)
    try:
        state = tmp['state']
    except KeyError:
        state = tmp['model_state']
    except:
        raise
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key and not 'gamma' in key and not 'beta' in key:
            newkey = key.replace("feature.", "")
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model.load_state_dict(state)
    model.eval()

    # save feature file
    print('  extract and save features...')
    if params.save_epoch != -1:
        featurefile = os.path.join(checkpoint_dir.replace(
            "checkpoints", "features"), split + "_" + str(params.save_epoch) + ".hdf5")
    else:
        featurefile = os.path.join(checkpoint_dir.replace(
            "checkpoints", "features"), split + ".hdf5")
    dirname = os.path.dirname(featurefile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    if 'evo' in params.method:
        # filter out the ft parameters
        feature_params = [i for i in tmp['feature_params'] if i.shape[0]
            != 1 or i.shape[2] != 1 or i.shape[3] != 1]
        save_features_patched(model, feature_params, data_loader, featurefile, device)
    else:
        save_features(model, data_loader, featurefile, device)

    print('\nStage 2: evaluate')
    acc_all = []
    iter_num = 1000
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    # model
    print('  build metric-based model')
    if params.method == 'protonet':
        model = ProtoNet(model_dict[params.model],
                         device=device, **few_shot_params)
    elif params.method == 'matchingnet':
        model = MatchingNet(
            model_dict[params.model], device=device, **few_shot_params)
    elif params.method == 'gnnnet':
        model = GnnNet(model_dict[params.model],
                       device=device, **few_shot_params)
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4':
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6':
            feature_model = backbone.Conv6NP
        else:
            feature_model = model_dict[params.model]
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model = RelationNet(
            feature_model, loss_type=loss_type, device=device, **few_shot_params)
    elif params.method in ['relationnet_evo', 'relationnet_softmax_evo']:
        if params.model == 'Conv4':
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6':
            feature_model = backbone.Conv6NP
        else:
            feature_model = model_dict[params.model]
        loss_type = 'mse' if params.method == 'relationnet_evo' else 'softmax'
        model = PatchedRelationNet(
            feature_model, loss_type=loss_type, device=device, **few_shot_params)
    else:
        raise ValueError('Unknown method')
    model = model.to(device=device)
    model.eval()

    # load model
    checkpoint_dir = '%s/checkpoints/%s' % (params.save_dir, params.name)
    if params.save_epoch != -1:
        modelfile = get_assigned_file(checkpoint_dir, params.save_epoch)
    else:
        modelfile = get_best_file(checkpoint_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile, map_location=device)
        try:
            model.load_state_dict(tmp['state'])
        except RuntimeError:
            print('warning! RuntimeError when load_state_dict()!')
            model.load_state_dict(tmp['state'], strict=False)
        except KeyError:
            for k in tmp['model_state']:  # revise latter
                if 'running' in k:
                    tmp['model_state'][k] = tmp['model_state'][k].squeeze()
            model.load_state_dict(tmp['model_state'], strict=False)
        except:
            raise
        if 'evo' in params.method:
            model.feature_params = tmp['feature_params']
            model.rm_params = tmp['rm_params']

    # load feature file
    print('  load saved feature file')
    cl_data_file = feat_loader.init_loader(featurefile)

    # start evaluate
    print('  evaluate')
    for i in range(iter_num):
        if 'evo' in params.method:
            acc = feature_evaluation_patched(
                cl_data_file, model, n_query=15, **few_shot_params)
        else:
            acc = feature_evaluation(
                cl_data_file, model, n_query=15, **few_shot_params)
        acc_all.append(acc)

    # statistics
    print('  get statistics')
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('  %d test iterations: Acc = %4.2f%% +- %4.2f%%' %
          (iter_num, acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))

    # save the statistics into a file
    experiment_update_dict = {'test_acc_mean': acc_mean, 'test_acc_std': acc_std}
    update_json_experiment_log_dict(experiment_update_dict, params)

    # remove feature files [optional]
    if remove_featurefile:
        os.remove(featurefile)
