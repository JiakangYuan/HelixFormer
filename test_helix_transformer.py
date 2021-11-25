import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
from utils.train_test import train_test_helix_transformer


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # dataset
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(
            dataset[0][0].shape, len(dataset), dataset.n_classes))
    if not args.sauc:
        n_way = 5
    else:
        n_way = 2

    print(n_way)
    n_shot, n_query = args.shot, 15
    n_batch = 500
    ep_per_batch = 4
    batch_sampler = CategoriesSampler(
            dataset.label, n_batch, n_way, n_shot + n_query,
            ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)

    if config.get('load'):
        embedding_net = models.load(torch.load(config['load']))
    else:
        embedding_net = models.make(config['encoder'], **config['encoder_args'])

        if config.get('load_encoder'):
            embedding_net = models.load(torch.load(config['load_encoder']))
            embedding_net = embedding_net.encoder

    if config.get('load_relation'):
        relation_net_sv = torch.load(config['load_relation'])
        relation_net = models.load(relation_net_sv, name='relation_net')

    if config.get('load_trans'):
        trans_net_sv = torch.load(config['load_trans'])
        trans_net = models.load(trans_net_sv, name='trans_net')

    if config.get('_parallel'):
        embedding_net = nn.DataParallel(embedding_net)
        relation_net = nn.DataParallel(relation_net)
        trans_net = nn.DataParallel(trans_net)

    embedding_net.eval()
    relation_net.eval()
    trans_net.eval()

    utils.log('num params: {}'.format(utils.compute_n_params(embedding_net)))

    # testing
    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}

    test_epochs = args.test_epochs
    np.random.seed(0)
    va_lst = []
    for epoch in range(1, test_epochs + 1):
        for data, _ in tqdm(loader, leave=False):
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=ep_per_batch)

            labels_query = fs.make_nk_label(n_way, n_query, ep_per_batch=ep_per_batch).cuda()
            labels_support = fs.make_nk_label(n_way, n_shot, ep_per_batch=ep_per_batch).cuda()

            with torch.no_grad():
                if not args.sauc:
                    acc, loss = train_test_helix_transformer(embedding_net, relation_net, trans_net, x_shot, x_query,
                                                             labels_query, n_way, n_shot, n_query, ep_per_batch)

                    aves['vl'].add(loss.item(), len(data))
                    aves['va'].add(acc, len(data))
                    va_lst.append(acc)

        print('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f} '.format(
            epoch, aves['va'].item() * 100,
            mean_confidence_interval(va_lst) * 100,
            aves['vl'].item(), _[-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_helix_transformer.yaml')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-epochs', type=int, default=5)
    parser.add_argument('--sauc', action='store_true')
    parser.add_argument('--gpu', default='5,6,7')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)
    main(config)

