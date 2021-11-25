import torch
import utils
import torch.nn as nn
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss().cuda()


def train_relation_based(embedding_net, relation_net, x_shot, x_query, labels_query, n_train_way, n_train_shot,
                            n_train_query, ep_per_batch, loss_type):

    # First, Reshape Operation for the following relation-based metric
    img_shape = x_shot.shape[-3:]
    x_shot = x_shot.view(-1, *img_shape)
    x_query = x_query.view(-1, *img_shape)

    embeds_all = embedding_net(torch.cat([x_shot, x_query], dim=0))

    embeds_support, embeds_query = embeds_all[:len(x_shot)], embeds_all[-len(x_query):]

    feature_dim = embeds_support.shape[-3:]
    extend_feature_dim = list(embeds_support.shape[-3:])
    extend_feature_dim[0] *= 2

    embeds_support = embeds_support.view(ep_per_batch, n_train_way, n_train_shot, *feature_dim).mean(dim=2)
    embeds_query = embeds_query.view(ep_per_batch, n_train_way * n_train_query, *feature_dim)

    # Second, Calculate relation
    # each query sample links to every samples to calculate relations
    # to form a 100x128 matrix for relation network
    embeds_support_ext = embeds_support.unsqueeze(1).repeat(1, n_train_way*n_train_query,1,1,1,1)
    embeds_query_ext = embeds_query.unsqueeze(1).repeat(1, n_train_way,1,1,1,1)
    embeds_query_ext = torch.transpose(embeds_query_ext, 1, 2)

    relation_pairs = torch.cat((embeds_support_ext, embeds_query_ext), 3).view(-1, *extend_feature_dim)
    relations = relation_net(relation_pairs).view(-1, n_train_way)

    if loss_type == 'softmax':
        loss = F.cross_entropy(relations, labels_query)
        acc = utils.compute_acc(relations, labels_query)

    return acc, loss


def train_test_helix_transformer(embedding_net, relation_net, trans_net, x_shot, x_query, labels_query, n_train_way, n_train_shot,
                                 n_train_query, ep_per_batch):
    img_shape = x_shot.shape[-3:]
    x_shot = x_shot.view(-1, *img_shape)
    x_query = x_query.view(-1, *img_shape)
    num_query = n_train_way * n_train_query

    embeds_all = embedding_net(torch.cat([x_shot, x_query], dim=0))

    embeds_support, embeds_query = embeds_all[:len(x_shot)], embeds_all[-len(x_query):]

    _, c, h, w = embeds_support.size()
    feature_dim = embeds_support.shape[-3:]
    extend_feature_dim = list(embeds_support.shape[-3:])
    extend_feature_dim[0] *= 2

    embeds_query = embeds_query.view(ep_per_batch, n_train_way * n_train_query, *feature_dim)
    embeds_support = embeds_support.view(ep_per_batch, n_train_way, n_train_shot, *feature_dim)

    embeds_support_ext = embeds_support.unsqueeze(1).repeat(1, n_train_way*n_train_query, 1, 1, 1, 1, 1)
    embeds_query_ext = embeds_query.unsqueeze(1).repeat(1, n_train_way, 1, 1, 1, 1)
    embeds_query_ext = embeds_query_ext.unsqueeze(2).repeat(1, 1, n_train_shot, 1, 1, 1, 1)
    embeds_query_ext = embeds_query_ext.permute(0,3,1,2,4,5,6).contiguous()
    embeds_support_ext = embeds_support_ext.view(-1, *feature_dim)
    embeds_query_ext = embeds_query_ext.view(-1, *feature_dim)
    embeds_support_ext, embeds_query_ext = trans_net(embeds_support_ext, embeds_query_ext)

    feature_dim = embeds_support_ext.shape[-3:]
    extend_feature_dim = list(embeds_support_ext.shape[-3:])
    extend_feature_dim[0] *= 2

    embeds_support_ext = embeds_support_ext.view(ep_per_batch, num_query, n_train_way, n_train_shot, *feature_dim)
    embeds_query_ext = embeds_query_ext.view(ep_per_batch, num_query, n_train_way, n_train_shot, *feature_dim)
    relation_pairs = torch.cat((embeds_support_ext, embeds_query_ext), 3).view(-1, *extend_feature_dim)
    relations = relation_net(relation_pairs).view(-1, n_train_way, n_train_shot)
    relations = relations.mean(dim=-1)

    loss = criterion(relations, labels_query)
    acc = utils.compute_acc(relations, labels_query)
    return acc, loss


def train_test_helix_transformer_resnet12(embedding_net, relation_net, trans_net, x_shot, x_query, labels_query,
                                             n_train_way, n_train_shot, n_train_query, ep_per_batch):
    img_shape = x_shot.shape[-3:]
    x_shot = x_shot.view(-1, *img_shape)
    x_query = x_query.view(-1, *img_shape)
    num_query = n_train_way * n_train_query

    # resnet12 backbone
    embeds_all = torch.cat([x_shot, x_query], dim=0)
    embeds_all = embedding_net.module.layer1(embeds_all)
    embeds_all = embedding_net.module.layer2(embeds_all)
    embeds_all = embedding_net.module.layer3(embeds_all)

    embeds_support, embeds_query = embeds_all[:len(x_shot)], embeds_all[-len(x_query):]

    _, c, h, w = embeds_support.size()

    feature_dim = embeds_support.shape[-3:]
    extend_feature_dim = list(embeds_support.shape[-3:])
    extend_feature_dim[0] *= 2

    embeds_query = embeds_query.view(ep_per_batch, n_train_way * n_train_query, *feature_dim)
    embeds_support = embeds_support.view(ep_per_batch, n_train_way, n_train_shot, *feature_dim)
    embeds_support_ext = embeds_support.unsqueeze(1).repeat(1, n_train_way*n_train_query,1,1,1,1,1)

    embeds_query_ext = embeds_query.unsqueeze(1).repeat(1, n_train_way,1,1,1,1)
    embeds_query_ext = embeds_query_ext.unsqueeze(2).repeat(1,1,n_train_shot, 1,1,1,1)
    embeds_query_ext = embeds_query_ext.permute(0,3,1,2,4,5,6).contiguous()

    embeds_support_ext = embeds_support_ext.view(-1, *feature_dim)
    embeds_query_ext = embeds_query_ext.view(-1, *feature_dim)
    embeds_support_ext, embeds_query_ext = trans_net(embeds_support_ext, embeds_query_ext)
    feature_dim = embeds_support_ext.shape[-3:]
    extend_feature_dim = list(embeds_support_ext.shape[-3:])
    extend_feature_dim[0] *= 2

    embeds_support_ext = embeds_support_ext.view(ep_per_batch, num_query, n_train_way, n_train_shot, *feature_dim)
    embeds_query_ext = embeds_query_ext.view(ep_per_batch, num_query, n_train_way, n_train_shot, *feature_dim)
    relation_pairs = torch.cat((embeds_support_ext, embeds_query_ext), 3).view(-1, *extend_feature_dim)
    relations = relation_net(relation_pairs).view(-1, n_train_way, n_train_shot)
    relations = relations.mean(dim=-1)

    loss = criterion(relations, labels_query)
    acc = utils.compute_acc(relations, labels_query)
    return acc, loss


