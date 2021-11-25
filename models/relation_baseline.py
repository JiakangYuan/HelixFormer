import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('relation-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={},
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query, n_train_way=5, n_train_shot=1):

        # First, Reshape Operation for the following relation-based metric
        img_shape = x_shot.shape[-3:]
        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)

        embeds_all = self.encoder(torch.cat([x_shot, x_query], dim=0))

        embeds_support, embeds_query = embeds_all[:len(x_shot)], embeds_all[-len(x_query):]

        n_q, c, h, w = embeds_query.size()
        n_s, _, _, _ = embeds_support.size()

        embeds_support = embeds_support.unsqueeze(0).expand(n_q, -1, -1, -1, -1).contiguous().view(n_q * n_s, c, h, w)

        embeds_support = embeds_support.view(n_q, n_s, c, h * w).permute(0, 2, 1, 3).contiguous().view(n_q, c,
                                                                                                       n_s * h * w)  # (n_q, c, n_s*h*w)
        embeds_support_norm = F.normalize(embeds_support, dim=1)

        embeds_query = embeds_query.view(n_q, c, h * w).permute(0, 2, 1)  # (n_q, h*w, c)
        embeds_query_norm = F.normalize(embeds_query, dim=2)

        match_score = torch.matmul(embeds_query_norm, embeds_support_norm)  # (n_q, h*w, n_s*h*w)
        match_score = match_score.view(n_q, h * w, n_s, h * w).permute(0, 2, 1, 3)  # (n_q, n_s, h*w, h*w)

        final_local_score = torch.sum(match_score.contiguous().view(n_q, n_s, h * w, h * w), dim=-1)

        final_score = torch.mean(final_local_score, dim=-1) * self.temp

        logits = torch.mean(final_score.view(-1, n_train_way, n_train_shot), dim=2)

        return logits

