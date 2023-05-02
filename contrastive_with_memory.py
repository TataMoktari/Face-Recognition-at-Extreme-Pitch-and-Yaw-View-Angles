import torch
from torch import nn
import math
import numpy as np


class ContrastMemory(nn.Module):

    def __init__(self, inputSize, K, T=0.07, momentum=0.5, base_temperature=0.07,
                 frontal_csv=None, profile_csv=None, device=None, resume=None, num_positive=1.):
        super(ContrastMemory, self).__init__()
        '''
        input_size = feature_dim
        output_size_profile = number of training samples which are profile
        output_size_frontal = number of training samples which are frontal
        K = number of negative to be chosen for every image
        T = temperature in SupContrastive loss
        momentum = momentum for updating memory
        '''

        output_size_profile = 604731
        output_size_frontal = 16084
        self.base_temperature = base_temperature
        self.nLem_p = output_size_profile
        self.unigrams_p = torch.ones(self.nLem_p)
        self.multinomial_p = AliasMethod(self.unigrams_p, device=device)
        self.multinomial_p.cuda()

        self.nLem_f = output_size_frontal
        self.unigrams_f = torch.ones(self.nLem_f)
        self.multinomial_f = AliasMethod(self.unigrams_f, device=device)
        self.multinomial_f.cuda()

        self.K = K  ##number of negative to be chosen for every image
        self.num_positive = num_positive

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1',
                             torch.rand(output_size_frontal, inputSize).mul_(2 * stdv).add_(-stdv))  # for frontal
        self.register_buffer('memory_v2',
                             torch.rand(output_size_profile, inputSize).mul_(2 * stdv).add_(-stdv))  # for profile

        # if resume:
        #     print('loading last memory from the saved model')
        #     state = torch.load('ch3/epoch15.pt',map_location=device)
        #     self.memory_v1.copy_(state['memory_frontal'])
        #     self.memory_v2.copy_(state['memory_profile'])

    def forward(self, v1, y1, v2, y2, idx1=None, idx2=None, opt=None):
        '''
        v1: frontal features (B,512)
        y1: frontal image index in the frontal csv (B,1)
        idx1 = one positive + K negative indexs (all images are frontal) (B,K+1)
        v2: profile features (B,512)
        y2: profile image index in the profile csv (B,1)
        idx2 = one positive + K negative indexs (all images are profile) (B,K+1)
        '''
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()

        momentum = self.params[3].item()
        batchSize = v1.size(0)
        output_size_frontal = self.memory_v1.size(0)
        output_size_profile = self.memory_v2.size(0)
        inputSize = self.memory_v1.size(1)

        # sample k negative frontal and one positive frontal from memory
        weight_v1 = torch.index_select(self.memory_v1, 0, idx1.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + self.num_positive, inputSize)
        # sample k negative profile and one positive profile from memory
        weight_v2 = torch.index_select(self.memory_v2, 0, idx2.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + self.num_positive, inputSize)
        # concat profile and frontal features of memory
        weight = torch.cat((weight_v1, weight_v2), dim=1)
        weight = weight.repeat(2, 1, 1)
        # concat profile and frontal features of model
        v = torch.cat((v1, v2), dim=0)

        # calculate contrastive loss
        anchor_dot_contrast = torch.div(torch.bmm(weight, v.view(2 * batchSize, inputSize, 1)), T)
        anchor_dot_contrast = anchor_dot_contrast.squeeze(2)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = torch.zeros((weight.shape[0], weight.shape[1] // 2))

        mask[:, self.num_positive - 1] = 1.
        mask = mask.repeat(1, 2).cuda()
        # logits_mask = 1. - mask

        # compute log_prob
        exp_logits = torch.exp(logits)  # * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(T / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(2, batchSize).mean()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y1.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y1, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y2.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y2, updated_v2)

        return loss


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs, device=None):
        self.device = device
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda(device=self.device)
        self.alias = self.alias.cuda(device=self.device)

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj

