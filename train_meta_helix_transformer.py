import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import setup_seed
import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
from utils import weights_init
from utils.train_test import train_test_helix_transformer


def main(config):
    save_name = args.name
    if save_name is None:
        save_name = 'Meta_train_helix-transformer_{}_2heads_{}-{}shot_1.12'.format(
            config['trans_net'], config['train_dataset'], config['n_shot'])
        save_name += '_' + config['encoder']
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    # Dataset

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('n_train_query') is not None:
        n_train_query = config['n_train_query']
    else:
        n_train_query = n_query
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    utils.log('train dataset: {} (x{}), {}'.format(
        train_dataset[0][0].shape, len(train_dataset),
        train_dataset.n_classes), filename='{}_meta-train.txt'.format(config['train_dataset']))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    train_sampler = CategoriesSampler(
        train_dataset.label, config['train_batches'],
        n_train_way, n_train_shot + n_train_query,
        ep_per_batch=ep_per_batch)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=0, pin_memory=False)

    # tval
    if config.get('tval_dataset'):
        tval_dataset = datasets.make(config['tval_dataset'],
                                     **config['tval_dataset_args'])
        utils.log('tval dataset: {} (x{}), {}'.format(
            tval_dataset[0][0].shape, len(tval_dataset),
            tval_dataset.n_classes), filename='{}_meta-train.txt'.format(config['train_dataset']))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
        tval_sampler = CategoriesSampler(
            tval_dataset.label, 200,
            n_way, n_shot + n_query,
            ep_per_batch=ep_per_batch)
        tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                 num_workers=0, pin_memory=False)
    else:
        tval_loader = None

    # val
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {} (x{}), {}'.format(
        val_dataset[0][0].shape, len(val_dataset),
        val_dataset.n_classes), filename='{}_meta-train.txt'.format(config['train_dataset']))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(
        val_dataset.label, 200,
        n_way, n_shot + n_query,
        ep_per_batch=ep_per_batch)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=0, pin_memory=False)

    # Model and optimizer

    if config.get('load'):
        model_sv = torch.load(config['load'])
        embedding_net = models.load(model_sv)
    else:
        embedding_net = models.make(config['encoder'], **config['encoder_args'])

        if config.get('load_encoder'):
            embedding_net = models.load(torch.load(config['load_encoder']))
            embedding_net = embedding_net.encoder

    if config.get('load_relation'):
        relation_net_sv = torch.load(config['load'])
        relation_net = models.load(relation_net_sv, name='relation_net')
    else:
        relation_net = models.make(config['relation_net'], **config['relation_net_args'])
        relation_net.apply(weights_init)

    if config.get('load_trans'):
        trans_net_sv = torch.load(config['load'])
        trans_net = models.load(trans_net_sv, name='trans_net')
    else:
        trans_net = models.make(config['trans_net'], **config['trans_net_args'])

    embedding_net = embedding_net.cuda()
    relation_net = relation_net.cuda()
    trans_net.cuda()

    if config.get('_parallel'):
        embedding_net = nn.DataParallel(embedding_net)
        relation_net = nn.DataParallel(relation_net)
        trans_net = nn.DataParallel(trans_net)

    utils.log('num params: {}'.format(utils.compute_n_params(embedding_net)),
              filename='{}_meta-train.txt'.format(config['train_dataset']))

    utils.log('num params: {}'.format(utils.compute_n_params(relation_net)),
              filename='{}_meta-train.txt'.format(config['train_dataset']))

    utils.log('num params: {}'.format(utils.compute_n_params(trans_net)),
              filename='{}_meta-train.txt'.format(config['train_dataset']))

    optimizer, lr_scheduler = utils.make_optimizer(
        embedding_net.parameters(),
        config['optimizer'], **config['optimizer_args'])

    optimizer_rn, lr_scheduler_rn = utils.make_optimizer(
        relation_net.parameters(),
        config['optimizer_rn'], **config['optimizer_rn_args'])

    optimizer_trans, lr_schedular_trans = utils.make_optimizer(
        trans_net.parameters(),
        config['optimizer_trans'], **config['optimizer_trans_args']
    )

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    for epoch in range(1, max_epoch + 1):
        torch.autograd.set_detect_anomaly(True)
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}
        # lr_scheduler.step()
        lr_scheduler.step()
        lr_schedular_trans.step()
        lr_scheduler_rn.step()

        # train
        embedding_net.train()
        relation_net.train()
        trans_net.train()

        if config.get('freeze_bn'):
            utils.freeze_bn(embedding_net)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        np.random.seed(epoch)
        for data, _ in tqdm(train_loader, desc='train', leave=False):
            x_shot, x_query = fs.split_shot_query(
                data.cuda(), n_train_way, n_train_shot, n_train_query,
                ep_per_batch=ep_per_batch)

            labels_query = fs.make_nk_label(n_train_way, n_train_query,
                                     ep_per_batch=ep_per_batch).cuda()  # (300, )

            labels_support = fs.make_nk_label(n_train_way, n_train_shot,
                                       ep_per_batch=ep_per_batch).cuda()

            acc, loss =  train_test_helix_transformer(embedding_net, relation_net, trans_net, x_shot,
                                                      x_query, labels_query, n_train_way, n_train_shot,
                                                      n_train_query, ep_per_batch)

            optimizer.zero_grad()
            optimizer_rn.zero_grad()
            optimizer_trans.zero_grad()

            loss.backward()

            optimizer.step()
            optimizer_rn.step()
            optimizer_trans.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            logits = None;
            loss = None

        embedding_net.eval()
        relation_net.eval()
        trans_net.eval()

        for name, loader, name_l, name_a in [
            ('tval', tval_loader, 'tvl', 'tva'),
            ('val', val_loader, 'vl', 'va')]:

            if (config.get('tval_dataset') is None) and name == 'tval':
                continue

            np.random.seed(0)
            for data, _ in tqdm(loader, desc=name, leave=False):
                x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=ep_per_batch)

                labels_query = fs.make_nk_label(n_way, n_query,
                                                ep_per_batch=ep_per_batch).cuda()  # (300, )

                labels_support = fs.make_nk_label(n_way, n_shot,
                                                  ep_per_batch=ep_per_batch).cuda()

                with torch.no_grad():
                    acc, loss = train_test_helix_transformer(embedding_net, relation_net, trans_net, x_shot,
                                                             x_query, labels_query, n_way, n_shot,
                                                             n_query, ep_per_batch)

                aves[name_l].add(loss.item())
                aves[name_a].add(acc)

        _sig = int(_[-1])

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train {:.4f}|{:.4f}; tval {:.4f}|{:.4f}, '
                  'val {:.4f}|{:.4f}; {} {}/{} (@{})'.format(
            epoch, aves['tl'], aves['ta'],
            aves['tvl'], aves['tva'],
            aves['vl'], aves['va'], t_epoch, t_used, t_estimate, _sig), filename='{}_meta-train.txt'.format(config['train_dataset']))

        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)

        if config.get('_parallel'):
            model_ = embedding_net.module
            relation_net_ = relation_net.module
            trans_net_ = trans_net.module
        else:
            model_ = embedding_net
            relation_net_ = relation_net
            trans_net_ = trans_net

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),

            'optimizer_rn': config['optimizer_rn'],
            'optimizer_rn_args': config['optimizer_rn_args'],
            'optimizer_rn_sd': optimizer_rn.state_dict(),

            'optimizer_trans': config['optimizer_trans'],
            'optimizer_trans_args': config['optimizer_trans_args'],
            'optimizer_trans_sd': optimizer_trans.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['encoder'],
            'model_args': config['encoder_args'],
            'model_sd': model_.state_dict(),

            'relation_net': config['relation_net'],
            'relation_net_args': config['relation_net_args'],
            'relation_net_sd': relation_net_.state_dict(),

            'trans_net': config['trans_net'],
            'trans_net_args': config['trans_net_args'],
            'trans_net_sd': trans_net_.state_dict(),

            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

        writer.flush()


if __name__ == '__main__':
    setup_seed(2021)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_helix_transformer.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0, 1')
    parser.add_argument('--eps', type=float, default=0.0,
                       help='epsilon of label smoothing')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    eps = args.eps
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)

