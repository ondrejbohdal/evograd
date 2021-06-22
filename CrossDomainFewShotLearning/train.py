import json
import numpy as np
import os
import random
import time
import torch
import tqdm
from data.datamgr import SetDataManager, SimpleDataManager
from options import parse_args, get_resume_file, load_warmup_state
from methods.LFTNet import LFTNet
from methods.LFTNetMetaEvo import LFTNetMetaEvo


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# training iterations


def train(base_datamgr, base_set, aux_iter, val_loader, model, start_epoch, stop_epoch, params, device):

    # for validation
    max_acc = 0
    total_it = 0
    create_json_experiment_log(params)
    start_time = time.time()

    # training
    with tqdm.tqdm(total=stop_epoch-start_epoch) as pbar_epochs:
        for epoch in range(start_epoch, stop_epoch):
            print('Epoch: ' + str(epoch))
            # randomly split seen domains to pseudo-seen and pseudo-unseen domains
            random_set = random.sample(base_set, k=2)
            ps_set = random_set[0]
            pu_set = random_set[1:]
            print('PS set: ' + str(ps_set))
            print('PU set: ' + str(pu_set))
            ps_loader = base_datamgr.get_data_loader(os.path.join(
                params.data_dir, ps_set, 'base.json'), aug=params.train_aug)
            pu_loader = base_datamgr.get_data_loader([os.path.join(
                params.data_dir, dataset, 'base.json') for dataset in pu_set], aug=params.train_aug)
            # train loop
            model.train()
            train_start_time = time.time()
            total_it = model.trainall_loop(epoch, ps_loader, pu_loader, aux_iter, total_it)
            max_memory_allocated = torch.cuda.max_memory_allocated()
            train_end_time = time.time()

            # validate
            model.eval()
            val_start_time = time.time()
            with torch.no_grad():
                acc = model.test_loop(val_loader)
            val_end_time = time.time()


            # save
            if acc > max_acc:
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                model.save(outfile, epoch)
            else:
                print('GG!! best accuracy {:f}'.format(max_acc))
            if ((epoch + 1) % params.save_freq == 0) or (epoch == stop_epoch - 1):
                outfile = os.path.join(params.checkpoint_dir,
                                    '{:d}.tar'.format(epoch + 1))
                model.save(outfile, epoch)
            experiment_update_dict = {'val_acc': acc,
                                      'epoch': epoch,
                                      'train_time': train_end_time-train_start_time,
                                      'val_time': val_end_time-val_start_time,
                                      'max_memory_allocated': max_memory_allocated}
            update_json_experiment_log_dict(experiment_update_dict, params)
            pbar_epochs.update(1)

    train_time = time.time() - start_time
    experiment_update_dict = {'total_train_time': train_time}
    update_json_experiment_log_dict(experiment_update_dict, params)
    return


def create_json_experiment_log(params):
    json_experiment_log_file_name = os.path.join(
        'results', params.name) + '.json'
    experiment_summary_dict = {'val_acc': [], 'test_acc_mean': [],
                               'test_acc_std': [], 'epoch': [],
                               'train_time': [], 'val_time': [],
                               'total_train_time': [], 'max_memory_allocated': []}
                               
    with open(json_experiment_log_file_name, 'w') as f:
        json.dump(experiment_summary_dict, fp=f)


def update_json_experiment_log_dict(experiment_update_dict, params):
    json_experiment_log_file_name = os.path.join(
        'results', params.name) + '.json'
    with open(json_experiment_log_file_name, 'r') as f:
        summary_dict = json.load(fp=f)

    for key in experiment_update_dict:
        summary_dict[key].append(experiment_update_dict[key])

    with open(json_experiment_log_file_name, 'w') as f:
        json.dump(summary_dict, fp=f)


# --- main function ---
if __name__ == '__main__':

    # set numpy random seed
    np.random.seed(10)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("use GPU", device)
    else:
        device = torch.device('cpu')
        print("use CPU")

    # parse argument
    params = parse_args('train')
    print('--- LFTNet training: {} ---\n'.format(params.name))
    print(params)

    # output and tensorboard dir
    params.tf_dir = '%s/log/%s' % (params.save_dir, params.name)
    params.checkpoint_dir = '%s/checkpoints/%s' % (
        params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader
    print('\n--- prepare dataloader ---')
    print('  train with multiple seen domains (unseen domain: {})'.format(params.testset))
    datasets = ['miniImagenet', 'cars', 'places', 'cub', 'plantae']
    datasets.remove(params.testset)
    val_file = os.path.join(params.data_dir, 'miniImagenet', 'val.json')

    # model
    print('\n--- build LFTNet model ---')
    if 'Conv' in params.model:
        image_size = 84
    else:
        image_size = 224

    n_query = params.n_query
    n_query_meta_train = n_query
    train_few_shot_params = dict(
        n_way=params.train_n_way, n_support=params.n_shot)
    base_datamgr = SetDataManager(
        image_size, n_query=n_query_meta_train,  **train_few_shot_params)
    aux_datamgr = SimpleDataManager(image_size, batch_size=16)
    aux_iter = iter(cycle(aux_datamgr.get_data_loader(os.path.join(
        params.data_dir, 'miniImagenet', 'base.json'), aug=params.train_aug)))
    test_few_shot_params = dict(
        n_way=params.test_n_way, n_support=params.n_shot)
    val_datamgr = SetDataManager(
        image_size, n_query=n_query, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    
    if 'evo' in params.method:
        model = LFTNetMetaEvo(params, tf_path=params.tf_dir, device=device,
                              n_model_candidates=params.n_model_candidates, temperature=params.temperature)
    else:
        model = LFTNet(params, tf_path=params.tf_dir, device=device)
        
    model.to(device=device)

    # resume training
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.resume != '':
        resume_file = get_resume_file(
            '%s/checkpoints/%s' % (params.save_dir, params.resume), params.resume_epoch)
        if resume_file is not None:
            start_epoch = model.resume(resume_file)
            print('  resume the training with at {} epoch (model file {})'.format(
                start_epoch, params.resume))
        else:
            raise ValueError('No resume file')
    # load pre-trained feature encoder
    else:
        if params.warmup == 'gg3b0':
            raise Exception(
                'Must provide pre-trained feature-encoder file using --warmup option!')
        model.model.feature.load_state_dict(load_warmup_state(
            '%s/checkpoints/%s' % (params.save_dir, params.warmup), params.method, device), strict=False)
    # this part needs to be done after loading the pretrained weights
    if 'evo' in params.method:
        model.define_parameters()

    # training
    print('\n--- start the training ---')
    train(base_datamgr, datasets, aux_iter, val_loader,
          model, start_epoch, stop_epoch, params, device)
