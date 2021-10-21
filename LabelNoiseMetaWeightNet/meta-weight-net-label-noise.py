import argparse
import copy
import json
import os
import random
import time
from operator import mul

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import tqdm
from higher.patch import buffer_sync, make_functional
from higher.utils import get_func_params
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import backbone
import resnet
from load_corrupted_data import CIFAR10, CIFAR100


def pad(x):
    return F.pad(x.unsqueeze(0),
                 (4, 4, 4, 4), mode='reflect').squeeze()


def build_dataset():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            pad,
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        train_data_meta = CIFAR10(
            root='data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR10(
            root='data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR10(root='data', train=False,
                            transform=test_transform, download=True)

    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root='data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR100(
            root='data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR100(root='data', train=False,
                             transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_loader, train_meta_loader, test_loader


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


def parseArgs():
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset (cifar10 [default] or cifar100)')
    parser.add_argument('--corruption_prob', type=float, default=0.4,
                        help='label noise')
    parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                        help='Type of corruption ("unif" or "flip" or "flip2").')
    parser.add_argument('--num_meta', type=int, default=1000)
    parser.add_argument('--epochs', default=60, type=int,
                        help='number of total epochs to run')
    # RN32 does 60 epochs, each with 500 iters
    parser.add_argument('--iters', default=30000, type=int,
                        help='number of total iters to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', '--batch-size', default=100, type=int,
                        help='mini-batch size (default: 100)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=True,
                        type=bool, help='nesterov momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--layers', default=28, type=int,
                        help='total number of layers (default: 28)')
    parser.add_argument('--widen-factor', default=10, type=int,
                        help='widen factor (default: 10)')
    parser.add_argument('--droprate', default=0, type=float,
                        help='dropout probability (default: 0.0)')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='whether to use standard augmentation (default: True)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--name', default='dbg', type=str,
                        help='name of experiment')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--prefetch', type=int, default=6,
                        help='Pre-fetching threads.')
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--meta_increase_c', type=int, default=1)
    parser.add_argument('--method', default='or2',
                        type=str, help='or2/evo/or2own')
    parser.add_argument('--name_of_args_json_file',
                        default=None, type=str, help='')
    parser.add_argument(
        '--n_model_candidates', default=2, type=int, help='Number of model candidates')
    parser.add_argument('--temperature', default=0.05,
                        type=float, help='Temperature for meta-evolution')
    parser.add_argument('--sigma', default=0.001, type=float,
                        help='Temperature for meta-evolution')
    parser.set_defaults(augment=True)
    args = parser.parse_args()

    if args.name_of_args_json_file is not None:
        args = extract_args_from_json(
            json_file_path=args.name_of_args_json_file, existing_args_dict=args)

    return args


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch):
    """
    Original way of training MWN using second-order update. We compare to this version in the paper.
    """
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    train_meta_loader_iter = iter(train_meta_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        input_var = to_var(inputs, requires_grad=False)
        target_var = to_var(targets, requires_grad=False).long()

        meta_model = build_model()
        meta_model.load_state_dict(model.state_dict())
        outputs = meta_model(input_var)
        cost = F.cross_entropy(outputs, target_var, reduction='none')
        cost_v = torch.reshape(cost, (len(cost), 1))
        v_lambda = vnet(cost_v.data)

        norm_c = torch.sum(v_lambda)
        if norm_c != 0:
            v_lambda_norm = v_lambda / norm_c
        else:
            v_lambda_norm = v_lambda

        l_f_meta = torch.sum(cost_v * v_lambda_norm)
        meta_model.zero_grad()
        grads = torch.autograd.grad(
            l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr * ((0.1 ** int(epoch >= 40)) *
                             (0.1 ** int(epoch >= 50)))  # For ResNet32
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        try:
            input_validation, target_validation = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            input_validation, target_validation = next(train_meta_loader_iter)

        input_validation_var = to_var(input_validation, requires_grad=False)
        target_validation_var = to_var(target_validation.type(
            torch.LongTensor), requires_grad=False).long()

        y_g_hat = meta_model(input_validation_var)
        l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
        prec_meta = accuracy(
            y_g_hat.data, target_validation_var.data, topk=(1,))[0]

        optimizer_c.zero_grad()
        l_g_meta.backward()
        optimizer_c.step()

        outputs = model(input_var)
        cost_w = F.cross_entropy(outputs, target_var, reduction='none')
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(outputs.data, target_var.data, topk=(1,))[0]

        with torch.no_grad():
            w_new = vnet(cost_v)
        norm_v = torch.sum(w_new)

        if norm_v != 0:
            w_v = w_new / norm_v
        else:
            w_v = w_new

        loss = torch.sum(cost_v * w_v)

        losses.update(loss.item(), inputs.size(0))
        meta_losses.update(l_g_meta.item(), inputs.size(0))
        top1.update(prec_train.item(), inputs.size(0))
        meta_top1.update(prec_meta.item(), inputs.size(0))

        optimizer_a.zero_grad()
        loss.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      loss=losses, meta_loss=meta_losses, top1=top1, meta_top1=meta_top1))
    return losses.avg, meta_losses.avg, top1.avg, meta_top1.avg


def train_or2_own(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch):
    """
    Modified way of training MWN using second-order update.
    In this case we use a simpler implementation using fast weights.
    However, both second-order ways give very similar results, consume the same amount of memory
    and have relatively similar runtimes.
    """
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    train_meta_loader_iter = iter(train_meta_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        input_var = to_var(inputs, requires_grad=False)
        target_var = to_var(targets, requires_grad=False).long()

        for weight in model.parameters():
            weight.fast = None

        outputs = model(input_var)
        cost = F.cross_entropy(outputs, target_var, reduction='none')
        cost_v = torch.reshape(cost, (len(cost), 1))
        v_lambda = vnet(cost_v.data)

        norm_c = torch.sum(v_lambda)
        if norm_c != 0:
            v_lambda_norm = v_lambda / norm_c
        else:
            v_lambda_norm = v_lambda

        l_f_meta = torch.sum(cost_v * v_lambda_norm)
        optimizer_a.zero_grad()
        grads = torch.autograd.grad(
            l_f_meta, model.parameters(), create_graph=True)
        meta_lr = args.lr * ((0.1 ** int(epoch >= 40)) *
                             (0.1 ** int(epoch >= 50)))  # For ResNet32
        for k, weight in enumerate(model.parameters()):
            weight.fast = weight - meta_lr * grads[k]
        del grads

        try:
            input_validation, target_validation = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            input_validation, target_validation = next(train_meta_loader_iter)

        input_validation_var = to_var(input_validation, requires_grad=False)
        target_validation_var = to_var(target_validation.type(
            torch.LongTensor), requires_grad=False).long()

        y_g_hat = model(input_validation_var)
        l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
        prec_meta = accuracy(
            y_g_hat.data, target_validation_var.data, topk=(1,))[0]

        optimizer_c.zero_grad()
        l_g_meta.backward()
        optimizer_c.step()

        for weight in model.parameters():
            weight.fast = None

        outputs = model(input_var)
        cost_w = F.cross_entropy(outputs, target_var, reduction='none')
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(outputs.data, target_var.data, topk=(1,))[0]

        with torch.no_grad():
            w_new = vnet(cost_v)
        norm_v = torch.sum(w_new)

        if norm_v != 0:
            w_v = w_new / norm_v
        else:
            w_v = w_new

        loss = torch.sum(cost_v * w_v)

        losses.update(loss.item(), inputs.size(0))
        meta_losses.update(l_g_meta.item(), inputs.size(0))
        top1.update(prec_train.item(), inputs.size(0))
        meta_top1.update(prec_meta.item(), inputs.size(0))

        optimizer_a.zero_grad()
        loss.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      loss=losses, meta_loss=meta_losses, top1=top1, meta_top1=meta_top1))
    return losses.avg, meta_losses.avg, top1.avg, meta_top1.avg


def train_evo(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch):
    """
    Train MWN using EvoGrad.
    """
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    train_meta_loader_iter = iter(train_meta_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        input_var = to_var(inputs, requires_grad=False)
        target_var = to_var(targets, requires_grad=False).long()

        model_patched = make_functional(model)
        # sync BN statistics
        buffer_sync(model, model_patched)
        model_parameter = [i.detach() for i in get_func_params(model)]
        n_model_candidates = args.n_model_candidates
        sigma = args.sigma
        temperature = args.temperature

        # create multiple model candidates by perturbing the weights
        theta_list = [[j + sigma * torch.sign(torch.randn_like(j))
                       for j in model_parameter] for i in range(n_model_candidates)]

        pred_list = [model_patched(input_var, params=theta)
                     for theta in theta_list]

        # calculate the losses of the model copies
        loss_list = [F.cross_entropy(
            pred, target_var, reduction='none') for pred in pred_list]

        cost_v_list = [torch.reshape(cost, (len(cost), 1))
                       for cost in loss_list]
        v_lambda_list = [vnet(cost_v.data) for cost_v in cost_v_list]

        norm_c_list = [torch.sum(v_lambda) for v_lambda in v_lambda_list]

        v_lambda_norm_list = [v_lambda / norm_c if norm_c !=
                              0 else v_lambda for v_lambda, norm_c in zip(v_lambda_list, norm_c_list)]

        l_f_meta_list = [torch.sum(cost_v * v_lambda_norm)
                         for cost_v, v_lambda_norm in zip(v_lambda_norm_list, cost_v_list)]

        # calculate the weights of model copies
        weights = torch.softmax(-torch.stack(l_f_meta_list)/temperature, 0)

        # merge the model copies
        theta_updated = [sum(map(mul, theta, weights))
                         for theta in zip(*theta_list)]

        # reset the meta-validation set data iterator if needed
        try:
            input_validation, target_validation = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            input_validation, target_validation = next(train_meta_loader_iter)

        input_validation_var = to_var(input_validation, requires_grad=False)
        target_validation_var = to_var(target_validation.type(
            torch.LongTensor), requires_grad=False).long()

        # update the meta-knowledge
        y_g_hat = model_patched(input_validation_var, params=theta_updated)
        l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
        prec_meta = accuracy(
            y_g_hat.data, target_validation_var.data, topk=(1,))[0]

        optimizer_c.zero_grad()
        l_g_meta.backward()
        optimizer_c.step()

        outputs = model(input_var)
        cost_w = F.cross_entropy(outputs, target_var, reduction='none')
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(outputs.data, target_var.data, topk=(1,))[0]

        with torch.no_grad():
            w_new = vnet(cost_v)
        norm_v = torch.sum(w_new)

        if norm_v != 0:
            w_v = w_new / norm_v
        else:
            w_v = w_new

        loss = torch.sum(cost_v * w_v)

        losses.update(loss.item(), inputs.size(0))
        meta_losses.update(l_g_meta.item(), inputs.size(0))
        top1.update(prec_train.item(), inputs.size(0))
        meta_top1.update(prec_meta.item(), inputs.size(0))

        optimizer_a.zero_grad()
        loss.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      loss=losses, meta_loss=meta_losses, top1=top1, meta_top1=meta_top1))
    return losses.avg, meta_losses.avg, top1.avg, meta_top1.avg


def test(model, test_loader):
    model.eval()
    losses_test = AverageMeter()
    top1_test = AverageMeter()

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            input_test_var = to_var(inputs, requires_grad=False)
            target_test_var = to_var(targets, requires_grad=False)

            # compute output
            with torch.no_grad():
                output_test = model(input_test_var)

            loss_test = F.cross_entropy(output_test, target_test_var)
            prec_test = accuracy(
                output_test.data, target_test_var.data, topk=(1,))[0]

            losses_test.update(
                loss_test.data.item(), input_test_var.size(0))
            top1_test.update(prec_test.item(), input_test_var.size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
        losses_test.avg, top1_test.avg))

    return top1_test.avg, losses_test.avg


def build_model():
    if args.method == 'or2':
        model = resnet.ResNet32(
            args.dataset == 'cifar10' and 10 or 100, width=args.width)
    elif args.method == 'or2own':
        backbone.BasicBlock.maml = True
        backbone.ResNet32.maml = True
        model = backbone.ResNet32(
            args.dataset == 'cifar10' and 10 or 100, width=args.width)
    else:
        model = backbone.ResNet32(
            args.dataset == 'cifar10' and 10 or 100, width=args.width)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * ((0.1 ** int(epoch >= 40)) *
                    (0.1 ** int(epoch >= 50)))  # For ResNet32
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_json_experiment_log(params):
    json_experiment_log_file_name = os.path.join(
        'results', params.name) + '.json'
    experiment_summary_dict = {'val_acc': [], 'test_acc': [], 'val_loss': [],
                               'epoch': [], 'train_time': [], 'val_time': [],
                               'total_train_time': [], 'max_memory_allocated': [],
                               'train_loss_avg': [], 'train_meta_loss_avg': [], 'train_acc_avg': [], 'meta_train_acc_avg': [],
                               'pytorch_total_params_learnable': [], 'pytorch_total_params_learnable_meta': []}

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


if __name__ == '__main__':
    global args, device
    args = parseArgs()
    create_json_experiment_log(args)

    use_cuda = True

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, train_meta_loader, test_loader = build_dataset()
    # create model
    model = build_model()

    if args.dataset == 'cifar10':
        num_classes = 10
    if args.dataset == 'cifar100':
        num_classes = 100

    if args.method == 'or2':
        vnet = resnet.VNet(1, 100 * args.meta_increase_c, 1).cuda()
        optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                      momentum=args.momentum, nesterov=args.nesterov,
                                      weight_decay=args.weight_decay)
        optimizer_c = torch.optim.SGD(vnet.params(), 1e-3,
                                      momentum=args.momentum, nesterov=args.nesterov,
                                      weight_decay=args.weight_decay)
        pytorch_total_params_learnable = sum(p.numel()
                                             for p in model.params() if p.requires_grad)
        pytorch_total_params_learnable_meta = sum(p.numel()
                                                  for p in vnet.params() if p.requires_grad)
    else:
        vnet = backbone.VNet(1, 100 * args.meta_increase_c, 1).cuda()
        optimizer_a = torch.optim.SGD(model.parameters(), args.lr,
                                      momentum=args.momentum, nesterov=args.nesterov,
                                      weight_decay=args.weight_decay)
        optimizer_c = torch.optim.SGD(vnet.parameters(), 1e-3,
                                      momentum=args.momentum, nesterov=args.nesterov,
                                      weight_decay=args.weight_decay)
        pytorch_total_params_learnable = sum(p.numel()
                                             for p in model.parameters() if p.requires_grad)
        pytorch_total_params_learnable_meta = sum(p.numel()
                                                  for p in vnet.parameters() if p.requires_grad)
    print('Number of all learnable parameters:')
    print(pytorch_total_params_learnable)

    print('Number of all learnable meta parameters:')
    print(pytorch_total_params_learnable_meta)

    torch.backends.cudnn.benchmark = True
    best_acc = 0
    start_time = time.time()
    with tqdm.tqdm(total=args.epochs) as pbar_epochs:
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer_a, epoch)
            train_start_time = time.time()
            if args.method == 'or2':
                train_loss_avg, train_meta_loss_avg, train_acc_avg, meta_train_acc_avg = train(train_loader, train_meta_loader, model,
                                                                                               vnet, optimizer_a, optimizer_c, epoch)
            elif args.method == 'or2own':
                train_loss_avg, train_meta_loss_avg, train_acc_avg, meta_train_acc_avg = train_or2_own(train_loader, train_meta_loader, model,
                                                                                                       vnet, optimizer_a, optimizer_c, epoch)
            elif args.method == 'evo':
                train_loss_avg, train_meta_loss_avg, train_acc_avg, meta_train_acc_avg = train_evo(
                    train_loader, train_meta_loader, model, vnet, optimizer_a, optimizer_c, epoch)
            else:
                raise ValueError('Method not implemented: ' + args.method)

            max_memory_allocated = torch.cuda.max_memory_allocated()
            train_end_time = time.time()

            val_start_time = time.time()
            test_acc, test_loss = test(model=model, test_loader=test_loader)
            val_end_time = time.time()
            if test_acc >= best_acc:
                best_acc = test_acc
            experiment_update_dict = {'val_acc': test_acc, 'val_loss': test_loss,
                                      'epoch': epoch,
                                      'train_time': train_end_time-train_start_time,
                                      'val_time': val_end_time-val_start_time,
                                      'max_memory_allocated': max_memory_allocated,
                                      'train_loss_avg': train_loss_avg, 'train_meta_loss_avg': train_meta_loss_avg,
                                      'train_acc_avg': train_acc_avg, 'meta_train_acc_avg': meta_train_acc_avg}
            update_json_experiment_log_dict(experiment_update_dict, args)
            pbar_epochs.update(1)

    print('Best accuracy: ', best_acc)

    # test the final model
    test_acc, _ = test(model=model, test_loader=test_loader)
    print('Final accuracy: ', test_acc)
    train_time = time.time() - start_time
    experiment_update_dict = {
        'total_train_time': train_time, 'test_acc': test_acc,
        'pytorch_total_params_learnable': pytorch_total_params_learnable, 'pytorch_total_params_learnable_meta': pytorch_total_params_learnable_meta}
    update_json_experiment_log_dict(experiment_update_dict, args)
