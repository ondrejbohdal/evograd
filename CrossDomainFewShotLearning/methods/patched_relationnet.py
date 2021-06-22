# This code is modified from https://github.com/floodsung/LearningToCompare_FSL

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from higher.patch import buffer_sync
from higher.patch import make_functional
from higher.utils import get_func_params
from tensorboardX import SummaryWriter

from methods import backbone


class PatchedRelationNet(nn.Module):
    def __init__(self, model_func,  n_way, n_support, tf_path=None, loss_type='mse', device=None, flatten=False, leakyrelu=False, change_way=True):
        super(PatchedRelationNet, self).__init__()

        self.device = device
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.feature = model_func(flatten=flatten, leakyrelu=leakyrelu)
        self.feat_dim = self.feature.final_feat_dim
        # some methods allow different_way classification during training and test
        self.change_way = change_way
        self.tf_writer = SummaryWriter(
            log_dir=tf_path) if tf_path is not None else None

        # loss function
        self.loss_type = loss_type  # 'softmax' or 'mse'
        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # metric function
        # relation net features are not pooled, so self.feat_dim is [dim, w, h]
        self.relation_module = RelationModule(self.feat_dim, 8, self.loss_type)
        self.method = 'RelationNet'
    
    def define_parameters(self):
        self.feature_params = get_func_params(self.feature)
        self.rm_params = get_func_params(self.relation_module)

    def set_forward(self, x, feat_params, rm_params, is_feature=False):
        # get features
        z_support, z_query = self.parse_feature(x, is_feature, feat_params)
        z_support = z_support.contiguous()
        z_proto = z_support.view(
            self.n_way, self.n_support, *self.feat_dim).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, *self.feat_dim)

        # get relations with metric function
        z_proto_ext = z_proto.unsqueeze(0).repeat(
            self.n_query * self.n_way, 1, 1, 1, 1)
        z_query_ext = z_query.unsqueeze(0).repeat(self.n_way, 1, 1, 1, 1)
        z_query_ext = torch.transpose(z_query_ext, 0, 1)
        extend_final_feat_dim = self.feat_dim.copy()
        extend_final_feat_dim[0] *= 2
        relation_pairs = torch.cat(
            (z_proto_ext, z_query_ext), 2).view(-1, *extend_final_feat_dim)

        patched_relation_module = make_functional(self.relation_module)
        buffer_sync(self.relation_module, patched_relation_module)
        if rm_params:
            relations = patched_relation_module(
                relation_pairs, params=rm_params).view(-1, self.n_way)
        else:
            relations = patched_relation_module(
                relation_pairs, params=self.rm_params).view(-1, self.n_way)
            if self.training:
                # update the buffers of the model
                # we do it only when training with the base model - not when doing meta-evolution
                # sync between patched_relation_module and self.relation_module - but now in the other way than previously
                buffer_sync_reverse(self.relation_module, patched_relation_module)
        return relations

    def set_forward_loss(self, x, feat_params=None, rm_params=None):
        y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))

        scores = self.set_forward(x, feat_params, rm_params)
        if self.loss_type == 'mse':
            y_oh = utils.one_hot(y, self.n_way)
            y_oh = y_oh.to(device=self.device)
            loss = self.loss_fn(scores, y_oh)
        else:
            y = y.to(device=self.device).long()
            loss = self.loss_fn(scores, y)
        return scores, loss

    def forward(self, x):
        # NOTE: this function is not expected to be used with patched parameters
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature, feat_params=None):
        x = x.to(device=self.device)
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(
                self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            patched_feature = make_functional(self.feature)
            buffer_sync(self.feature, patched_feature)
            if feat_params:
                z_all = patched_feature(x, params=feat_params)
            else:
                z_all = patched_feature(x, params=self.feature_params)
                if self.training:
                    # update the buffers of the model
                    # we do it only when training with the base model - not when doing meta-evolution
                    # sync between patched_feature and self.feature - but now in the other way than previously
                    buffer_sync_reverse(self.feature, patched_feature)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):
        scores, loss = self.set_forward_loss(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query), loss.item()*len(y_query)

    def train_loop(self, epoch, train_loader, optimizer, total_it):
        # NOTE: this function is not expected to be used with patched parameters
        print_freq = len(train_loader) // 10
        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            _, loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

            if (i + 1) % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch,
                                                                        i + 1, len(train_loader), avg_loss/float(i+1)))
            if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar(
                    self.method + '/query_loss', loss.item(), total_it + 1)
            total_it += 1
        return total_it

    def test_loop(self, test_loader, record=None):
        loss = 0.
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this, loss_this = self.correct(x)
            acc_all.append(correct_this / count_this*100)
            loss += loss_this
            count += count_this

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('--- %d Loss = %.6f ---' % (iter_num,  loss/count))
        print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' %
              (iter_num,  acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))

        return acc_mean

# --- Convolution block used in the relation module ---


class RelationConvBlock(nn.Module):
    maml = False

    def __init__(self, indim, outdim, padding=0):
        super(RelationConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = backbone.Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = backbone.BatchNorm2d_fw(
                outdim, momentum=1, track_running_stats=False)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = backbone.BatchNorm2d_fix(
                outdim, momentum=1, track_running_stats=False)
            # self.BN = nn.BatchNorm2d(
            #     outdim, momentum=1, affine=True, track_running_stats=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

        for layer in self.parametrized_layers:
            backbone.init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out

# --- Relation module adopted in RelationNet ---


class RelationModule(nn.Module):
    maml = False

    def __init__(self, input_size, hidden_size, loss_type='mse'):
        super(RelationModule, self).__init__()

        self.loss_type = loss_type
        # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling
        padding = 1 if (input_size[1] < 10) and (input_size[2] < 10) else 0

        self.layer1 = RelationConvBlock(
            input_size[0]*2, input_size[0], padding=padding)
        self.layer2 = RelationConvBlock(
            input_size[0], input_size[0], padding=padding)

        def shrink_s(s): return int(
            (int((s - 2 + 2*padding)/2)-2 + 2*padding)/2)

        if self.maml:
            self.fc1 = backbone.Linear_fw(
                input_size[0] * shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size)
            self.fc2 = backbone.Linear_fw(hidden_size, 1)
        else:
            self.fc1 = nn.Linear(
                input_size[0] * shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size)
            self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.loss_type == 'mse':
            out = torch.sigmoid(self.fc2(out))
        elif self.loss_type == 'softmax':
            out = self.fc2(out)

        return out

def buffer_sync_reverse(module, fmodule, device=None):
    r"""One off sync (copy) of buffers in ``fmodule`` with those from ``module`` - reversed.
    """
    for key, value in fmodule._buffers.items():
        if not _torch.is_tensor(value):
            module._buffers[key] = value
        elif device is None:
            module._buffers[key] = value.clone().detach()
        else:
            module._buffers[key] = value.clone().detach().to(device)

    for name, child in fmodule._modules.items():
        if name in module._modules:
            buffer_sync(child, module._modules[name], device)
        else:
            raise KeyError(
                "Did not find expected submodule "
                "{} of monkey-patched module {}.".format(name, module)
            )
