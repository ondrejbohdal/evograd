from operator import mul

import torch
import torch.nn as nn
from higher.patch import buffer_sync
from higher.patch import make_functional
from higher.utils import get_func_params
from tensorboardX import SummaryWriter

from methods import backbone, patched_relationnet
from methods.backbone import model_dict


class LFTNetMetaEvo(nn.Module):
    def __init__(self, params, tf_path=None, change_way=True, device=None, n_model_candidates=2, temperature=0.05):
        super(LFTNetMetaEvo, self).__init__()

        self.device = device

        # tf writer
        self.tf_writer = SummaryWriter(
            log_dir=tf_path) if tf_path is not None else None

        # get metric-based model and enable L2L(maml) training
        train_few_shot_params = dict(
            n_way=params.train_n_way, n_support=params.n_shot)
        backbone.FeatureWiseTransformation2d_fix.feature_augment = True
        # we will not use fast weights in this approach
        self.n_model_candidates = n_model_candidates
        self.temperature = temperature

        if params.method in ['relationnet_evo', 'relationnet_softmax_evo']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            else:
                feature_model = model_dict[params.model]
            loss_type = 'mse' if params.method == 'relationnet_evo' else 'softmax'
            model = patched_relationnet.PatchedRelationNet(
                feature_model, loss_type=loss_type, tf_path=params.tf_dir, device=self.device, **train_few_shot_params)
        else:
            raise ValueError('Only RelationNet currently supported')

        self.model = model
        print('  train with {} framework'.format(params.method))

        # for auxiliary training
        feat_dim = self.model.feat_dim[0] if type(
            self.model.feat_dim) is list else self.model.feat_dim
        self.aux_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 64))
        self.aux_loss_fn = nn.CrossEntropyLoss()

        # total epochs
        self.total_epoch = params.stop_epoch

    def define_parameters(self):
        self.model.define_parameters()
        
        # optimizer
        # filter the ft parameters based on shape: [1, x, 1, 1]
        feat_params = [i for i in self.model.feature_params if i.shape[0]
                        != 1 or i.shape[2] != 1 or i.shape[3] != 1]
        rm_params = self.model.rm_params
        ft_params = [i for i in self.model.feature_params if i.shape[0]
                        == 1 and i.shape[2] == 1 and i.shape[3] == 1]
        self.model_optim = torch.optim.Adam(feat_params + rm_params + list(self.aux_classifier.parameters()))
        self.ft_optim = torch.optim.Adam(ft_params, weight_decay=1e-8, lr=1e-3)

    # split the parameters of feature-wise transforamtion layers and others
    def split_model_parameters(self):
        model_params = []
        ft_params = []
        for n, p in self.model.named_parameters():
            n = n.split('.')
            if n[-1] == 'gamma' or n[-1] == 'beta':
                ft_params.append(p)
            else:
                model_params.append(p)
        return model_params, ft_params

    def forward_aux_loss(self, x, y):
        x = x.to(device=self.device)
        y = y.to(device=self.device)
        patched_feature = make_functional(self.model.feature)
        buffer_sync(self.model.feature, patched_feature)
        feat = patched_feature(x, params=self.model.feature_params)
        if feat.dim() > 2:
            feat = nn.functional.avg_pool2d(feat, 7)
            feat = feat.view(feat.size(0), -1)
        scores = self.aux_classifier(feat)
        loss = self.aux_loss_fn(scores, y)
        return loss

    # jotinly train the model and the feature-wise transformation layers
    def trainall_loop(self, epoch, ps_loader, pu_loader, aux_iter, total_it):
        print_freq = len(ps_loader) / 10
        avg_model_loss = 0.
        avg_ft_loss = 0.
        print('Start of epoch ' + str(epoch))

        for i, ((x, _), (x_new, _)) in enumerate(zip(ps_loader, pu_loader)):

            # auxiliary loss for stabilize the training
            self.model.eval()
            # this statement also changes the training state of the patched feature, feature, etc.
            x_aux, y_aux = next(aux_iter)
            aux_loss = self.forward_aux_loss(x_aux, y_aux)
            aux_loss_weighted = aux_loss*(0.9**((20*epoch//self.total_epoch)))

            # classification loss with ft layers (optimize model)
            self.model.train()
            self.model.n_query = x.size(1) - self.model.n_support
            if self.model.change_way:
                self.model.n_way = x.size(0)

            # update of the base model
            _, model_loss = self.model.set_forward_loss(x)
            self.model_optim.zero_grad()
            aux_loss_weighted.backward()
            model_loss.backward()
            self.model_optim.step()

            # meta-learning of the ft layers
            sigma = self.model_optim.param_groups[0]['lr']
            # remove ft params based on their unique shape [1, x, 1, 1] - these will not be changed
            theta_list = [[j if j.shape[0] == 1 and j.shape[2] == 1 and j.shape[3] == 1 else j.detach() + sigma*torch.sign(torch.randn_like(j))
                           for j in self.model.feature_params] for i in range(self.n_model_candidates)]
            rm_list = [[j.detach() + sigma*torch.sign(torch.randn_like(j))
                        for j in self.model.rm_params] for i in range(self.n_model_candidates)]

            loss_list = [self.model.set_forward_loss(
                x,  feat_params=theta_param, rm_params=rm_param)[1] for theta_param, rm_param in zip(theta_list, rm_list)]
            
            weights = torch.softmax(-torch.stack(loss_list)/self.temperature, 0)
            
            theta_updated = [sum(map(mul, theta, weights))
                             for theta in zip(*theta_list)]
            rm_updated = [sum(map(mul, rm_param, weights))
                          for rm_param in zip(*rm_list)]
            
            # now use pseudo-unseen data
            self.model.eval()
            _, ft_loss = self.model.set_forward_loss(
                x_new, feat_params=theta_updated, rm_params=rm_updated)
            # optimize ft
            self.ft_optim.zero_grad()
            ft_loss.backward()
            self.ft_optim.step()
            
            # loss
            avg_model_loss += model_loss.item()
            avg_ft_loss += ft_loss.item()

            if (i + 1) % print_freq == 0:
                print('Memory usage: ' + str(torch.cuda.max_memory_allocated()))
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | model_loss {:f}, ft_loss {:f}'.format(
                    epoch + 1, self.total_epoch, i + 1, len(ps_loader), avg_model_loss/float(i+1), avg_ft_loss/float(i+1)))
            if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar(
                    'LFTNetMetaEvo/model_loss', model_loss.item(), total_it + 1)
                self.tf_writer.add_scalar(
                    'LFTNetMetaEvo/ft_loss', ft_loss.item(), total_it + 1)
                self.tf_writer.add_scalar(
                    'LFTNetMetaEvo/aux_loss', aux_loss.item(), total_it + 1)
            total_it += 1

        return total_it

    # train the model itself (with ft layers)
    def train_loop(self, epoch, base_loader, total_it):
        # NOTE this is not used
        print_freq = len(base_loader) / 10
        avg_model_loss = 0.

        # enable ft layers
        self.model.train()

        # training loop
        for i, (x, _) in enumerate(base_loader):

            # loss = model_loss
            self.model.n_query = x.size(1) - self.model.n_support
            if self.model.change_way:
                self.model.n_way = x.size(0)
            _, model_loss = self.model.set_forward_loss(x)

            # optimize
            self.model_optim.zero_grad()
            model_loss.backward()
            self.model_optim.step()

            # loss
            avg_model_loss += model_loss.item()
            if (i + 1) % print_freq == 0:
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | model_loss {:f}'.format(
                    epoch + 1, self.total_epoch, i + 1, len(base_loader), avg_model_loss/float(i+1)))
            if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar(
                    'LFTNetMetaEvo/model_loss', model_loss.item(), total_it + 1)
            total_it += 1
        return total_it

    def test_loop(self, test_loader, record=None):
        self.model.eval()
        return self.model.test_loop(test_loader, record)

    def to(self, device=None):
        self.model.to(device=device)
        self.aux_classifier.to(device=device)

    def reset(self, warmUpState=None):

        # reset feature
        if warmUpState is not None:
            self.model.feature.load_state_dict(warmUpState, strict=False)
            print('    reset feature success!')

        # reset other module
        self.model.reset_modules()
        self.model.to(device=self.device)

        # reset optimizer
        feat_params = [i for i in self.model.feature_params if i.shape[0]
                        != 1 or i.shape[2] != 1 or i.shape[3] != 1]
        rm_params = self.model.rm_params
        ft_params = [i for i in self.model.feature_params if i.shape[0]
                        == 1 and i.shape[2] == 1 and i.shape[3] == 1]
        self.model_optim = torch.optim.Adam(feat_params + rm_params)
        return

    # save function
    def save(self, filename, epoch):
        state = {'epoch': epoch,
                 'model_state': self.model.state_dict(),
                 'aux_classifier_state': self.aux_classifier.state_dict(),
                 'feature_params': self.model.feature_params,
                 'rm_params': self.model.rm_params,
                 'model_optim_state': self.model_optim.state_dict(),
                 'ft_optim_state': self.ft_optim.state_dict()}
        torch.save(state, filename)

    # load function
    def resume(self, filename, device):
        state = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.aux_classifier.load_state_dict(state['aux_classifier_state'])
        self.model.feature_params = state['feature_params']
        self.model.rm_params = state['rm_params']
        self.model_optim.load_state_dict(state['model_optim_state'])
        self.ft_optim.load_state_dict(state['ft_optim_state'])
        return state['epoch'] + 1
