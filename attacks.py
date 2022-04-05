import torch
from torch import autograd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
TARGET_MULT = 10000.0

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

CARTPOLE_STD=[0.7322321, 1.0629482, 0.12236707, 0.43851405]
ACROBOT_STD=[0.36641926, 0.65119815, 0.6835106, 0.67652863, 2.0165246, 3.0202584]


def pgd(model, X, y, verbose=False, params={}, env_id="", norm_type='l_2'):
    X /= 255
    epsilon = params.get('epsilon', 0.00392)
    niters = params.get('niters', 10)
    img_min = params.get('img_min', 0.0)
    img_max = params.get('img_max', 1.0)
    network_type = params.get('network_type', 'nature')
    loss_func = params.get('loss_func', nn.CrossEntropyLoss())
    step_size = epsilon * 1.0 / niters
    y = Variable(torch.tensor(y))
    if verbose:
        print('epislon: {}, step size: {}, target label: {}'.format(epsilon, step_size, y))

    X_adv = Variable(X.data, requires_grad=True)

    for i in range(niters):

        if network_type == 'noisynet':
            model.model.sample()

        _, logits = model.forward_requires_grad(X_adv, return_q=True)

        loss = loss_func(logits, y)
        if verbose:
            print('current loss: ', loss.data.cpu().numpy())
        model.zero_grad()
        loss.backward()

        if norm_type == 'l_inf':
            eta = step_size * X_adv.grad.data.sign()
        elif norm_type == 'l_2':
            if not torch.norm(X_adv.grad).item():
                eta = step_size * X_adv.grad.data
            else:
                eta = step_size * X_adv.grad.data / torch.norm(X_adv.grad).data

        X_adv = Variable(X_adv.data + eta, requires_grad=True)
        # adjust to be within [-epsilon, epsilon]

        if norm_type == 'l_inf':
            eta = torch.clamp(X_adv.data - X.data, -epsilon, epsilon)

        elif norm_type == 'l_2':
            eta = X_adv.data - X.data
            # print('iter', i, 'second eta', torch.norm(eta))
            if torch.norm(eta) > epsilon:
                eta *= epsilon / torch.norm(eta)

        X_adv.data = X.data + eta
        if verbose:
            print('max eta: ', np.max(abs(eta.data.cpu().numpy())))
            print('linf diff before clamp: ', np.max(abs(X_adv.data.cpu().numpy()-X.data.cpu().numpy())))

        X_adv.data = torch.clamp(X_adv.data, img_min, img_max)
        if verbose:
            print('linf diff after clamp: ',np.max(abs(X_adv.data.cpu().numpy()-X.data.cpu().numpy())))

    if verbose:
        print('{} iterations'.format(i+1))

    return torch.clamp((X_adv.data * 255).long(), 0, 255)


def fgsm(model, X, y, verbose=False, params={}):
    epsilon=params.get('epsilon', 1)
    img_min=params.get('img_min', 0.0)
    img_max=params.get('img_max', 1.0)
    X_adv = Variable(X.data, requires_grad=True)
    logits = model.forward(X_adv)
    loss = F.nll_loss(logits, y)
    model.features.zero_grad()
    loss.backward()
    eta = epsilon*X_adv.grad.data.sign()
    X_adv = Variable(X_adv.data + eta, requires_grad=True)
    X_adv.data = torch.clamp(X_adv.data, img_min, img_max)
    return X_adv.data



def rand_attack(model, X, y, verbose=False, params={}, env_id=""):
    epsilon = params.get('epsilon', 0.00392)
    if env_id == "CartPole-v0":
        epsilon = torch.from_numpy(CARTPOLE_STD) * epsilon
    if env_id == "Acrobot-v1":
        epsilon = torch.from_numpy(ACROBOT_STD) * epsilon
    img_min = params.get('img_min', 0.0)
    img_max = params.get('img_max', 1.0)
    noise = 2 * epsilon * torch.rand(X.data.size()) - epsilon
    if USE_CUDA:
        noise = noise.cuda()
    X_adv = torch.clamp(X.data + noise, img_min, img_max)
    X_adv = Variable(X_adv.data, requires_grad=True)
    return X_adv.data


def attack(model, X, attack_config, loss_func=nn.CrossEntropyLoss(), epsilon=0.00392, smooth_type='', network_type='nature'):
    # method = attack_config.get('method', 'pgd')
    # verbose = attack_config.get('verbose', False)
    # params = attack_config.get('params', {})
    method = 'pgd'
    verbose = False
    params = {
        'epsilon': epsilon,
        'network_type': network_type,
    }
    params['loss_func'] = loss_func

    if network_type == 'noisynet':
        model.model.sample()

    if smooth_type == 'local':
        _, output = model.forward(X, cert=False, return_q=True)
    elif smooth_type == 'global':
        _, output = model.forward(X, return_q=True)
    else:
        raise NotImplementedError(f'smooth_type = {smooth_type} not implemented!')

    y = torch.argmax(output, dim=1)
    # y = model.act(X, cert=False)
    if method == 'cw':
        atk = cw
    elif method == 'rand':
        atk = rand_attack
    elif method == 'fgsm':
        atk = fgsm
    else:
        atk = pgd
    adv_X = atk(model, X, y, verbose=verbose, params=params)
    abs_diff = abs(adv_X.cpu().numpy()-X.cpu().numpy())
    if verbose:
        print('adv image range: {}-{}, ori action: {}, adv action: {}, l1 norm: {}, l2 norm: {}, linf norm: {}'.format(torch.min(adv_X).cpu().numpy(), torch.max(adv_X).cpu().numpy(), model.act(X)[0], model.act(adv_X)[0], np.sum(abs_diff), np.linalg.norm(abs_diff), np.max(abs_diff)))
    return adv_X

