import torch
import torch.nn as nn
from methods import protonet
from methods import matchingnet
from methods import relationnet
from methods import gnnnet
from methods.backbone import model_dict
from methods import backbone
from methods import gnn
from tensorboardX import SummaryWriter


class LFTNet(nn.Module):
    def __init__(self, params, tf_path=None, change_way=True, device=None):
        super(LFTNet, self).__init__()

        self.device = device

        # tf writer
        self.tf_writer = SummaryWriter(
            log_dir=tf_path) if tf_path is not None else None

        # get metric-based model and enable L2L(maml) training
        train_few_shot_params = dict(
            n_way=params.train_n_way, n_support=params.n_shot)
        backbone.FeatureWiseTransformation2d_fw.feature_augment = True
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.ResNet.maml = True
        if params.method == 'protonet':
            model = protonet.ProtoNet(
                model_dict[params.model], tf_path=params.tf_dir, device=self.device, **train_few_shot_params)
        elif params.method == 'matchingnet':
            backbone.LSTMCell.maml = True
            model = matchingnet.MatchingNet(
                model_dict[params.model], tf_path=params.tf_dir, device=self.device, **train_few_shot_params)
        elif params.method in ['relationnet', 'relationnet_softmax']:
            relationnet.RelationConvBlock.maml = True
            relationnet.RelationModule.maml = True
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            else:
                feature_model = model_dict[params.model]
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
            model = relationnet.RelationNet(
                feature_model, loss_type=loss_type, tf_path=params.tf_dir, device=self.device, **train_few_shot_params)
        elif params.method == 'gnnnet':
            gnnnet.GnnNet.maml = True
            gnn.Gconv.maml = True
            gnn.Wcompute.maml = True
            model = gnnnet.GnnNet(
                model_dict[params.model], tf_path=params.tf_dir, device=self.device, **train_few_shot_params)
        else:
            raise ValueError('Unknown method')
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

        # optimizer
        model_params, ft_params = self.split_model_parameters()
        self.model_optim = torch.optim.Adam(
            model_params + list(self.aux_classifier.parameters()))
        self.ft_optim = torch.optim.Adam(ft_params, weight_decay=1e-8, lr=1e-3)

        # total epochs
        self.total_epoch = params.stop_epoch

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
        feat = self.model.feature(x)
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

        for i, ((x, _), (x_new, _)) in enumerate(zip(ps_loader, pu_loader)):

            # clear fast weight
            for weight in self.split_model_parameters()[0]:
                weight.fast = None

            # auxiliary loss for stablize the training
            self.model.eval()
            x_aux, y_aux = next(aux_iter)
            aux_loss = self.forward_aux_loss(x_aux, y_aux)
            aux_loss_weighted = aux_loss*(0.9**((20*epoch//self.total_epoch)))

            # classifcation loss with ft layers (optimize model)
            self.model.train()
            self.model.n_query = x.size(1) - self.model.n_support
            if self.model.change_way:
                self.model.n_way = x.size(0)
            _, model_loss = self.model.set_forward_loss(x)

            # update model parameters according to model_loss
            meta_grad = torch.autograd.grad(
                model_loss, self.split_model_parameters()[0], create_graph=True)
            for k, weight in enumerate(self.split_model_parameters()[0]):
                weight.fast = weight - \
                    self.model_optim.param_groups[0]['lr']*meta_grad[k]
            meta_grad = [g.detach() for g in meta_grad]

            # classification loss with updated model and without ft layers (optimize ft layers)
            self.model.eval()
            _, ft_loss = self.model.set_forward_loss(x_new)

            # optimize model
            self.model_optim.zero_grad()
            aux_loss_weighted.backward()
            for k, weight in enumerate(self.split_model_parameters()[0]):
                weight.grad = meta_grad[k] if weight.grad is None else weight.grad + meta_grad[k]
            self.model_optim.step()

            # optimize ft
            self.ft_optim.zero_grad()
            ft_loss.backward()
            self.ft_optim.step()

            # loss
            avg_model_loss += model_loss.item()
            avg_ft_loss += ft_loss.item()

            if (i + 1) % print_freq == 0:
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | model_loss {:f}, ft_loss {:f}'.format(
                    epoch + 1, self.total_epoch, i + 1, len(ps_loader), avg_model_loss/float(i+1), avg_ft_loss/float(i+1)))
            if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar(
                    'LFTNet/model_loss', model_loss.item(), total_it + 1)
                self.tf_writer.add_scalar(
                    'LFTNet/ft_loss', ft_loss.item(), total_it + 1)
                self.tf_writer.add_scalar(
                    'LFTNet/aux_loss', aux_loss.item(), total_it + 1)
            total_it += 1

        return total_it

    # train the model itself (with ft layers)
    def train_loop(self, epoch, base_loader, total_it):
        print_freq = len(base_loader) / 10
        avg_model_loss = 0.

        # clear fast weight and enable ft layers
        self.model.train()
        for weight in self.model.parameters():
            weight.fast = None

        # trainin loop
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
                    'LFTNet/model_loss', model_loss.item(), total_it + 1)
            total_it += 1
        return total_it

    def test_loop(self, test_loader, record=None):
        self.model.eval()
        for weight in self.model.parameters():
            weight.fast = None
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
        self.model_optim = torch.optim.Adam(self.split_model_parameters()[0])
        return

    # save function
    def save(self, filename, epoch):
        state = {'epoch': epoch,
                 'model_state': self.model.state_dict(),
                 'aux_classifier_state': self.aux_classifier.state_dict(),
                 'model_optim_state': self.model_optim.state_dict(),
                 'ft_optim_state': self.ft_optim.state_dict()}
        torch.save(state, filename)

    # load function
    def resume(self, filename, device):
        state = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.aux_classifier.load_state_dict(state['aux_classifier_state'])
        self.model_optim.load_state_dict(state['model_optim_state'])
        self.ft_optim.load_state_dict(state['ft_optim_state'])
        return state['epoch'] + 1
