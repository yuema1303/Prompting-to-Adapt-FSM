import collections
import os

import torch
import torch.nn.functional as F
import utils


def _get_sub_policies(augment_id_list, magnitude_id_list, args):
    policies = []
    for n in range(args.subpolicy_num):    
        sub_policy = {}
        for i in range(args.op_num_pre_subpolicy): 
            policy = {}
            policy['op'] = args.augment_types[augment_id_list[n + i]]
            policy['magnitude'] = args.magnitude_types[magnitude_id_list[n + i]]
            sub_policy[i] = policy
        policies.append(sub_policy)
    return policies

class Controller(torch.nn.Module):
    # https://github.com/carpedm20/ENAS-pytorch/blob/master/models/controller.py
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args

        self.num_tokens = [len(args.augment_types),   
                           len(args.magnitude_types),    
                            ] * self.args.op_num_pre_subpolicy * self.args.subpolicy_num
        num_total_tokens = sum(self.num_tokens)     # 30个

        self.encoder = torch.nn.Embedding(num_total_tokens,
                                          args.controller_hid_size)
        self.lstm = torch.nn.LSTMCell(args.controller_hid_size, args.controller_hid_size)

        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(args.controller_hid_size, size)
            self.decoders.append(decoder)
        self._decoders = torch.nn.ModuleList(self.decoders)

        self._init_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.args.controller_hid_size),
                self.args.cuda,
                requires_grad=False)
        self.static_inputs = utils.keydefaultdict(_get_default_hidden)


    def _init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.controller_hid_size)
        return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False))

    def forward(self,  # pylint:disable=arguments-differ
                inputs,
                hidden,
                token_idx,
                is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[token_idx](hx)
        logits /= self.args.softmax_temperature

        if self.args.mode == 'train':
            logits = (self.args.tanh_c * torch.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1):
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        inputs = self.static_inputs[batch_size].cuda()
        hidden = self.static_init_hidden[batch_size]

        self.entropies = []
        self.log_probs = []
        policy_id_list = []
        magnitude_id_list = []
        prob_id_list = []

        for id in range(len(self.num_tokens)):
            logits, hidden = self.forward(inputs, hidden, id, is_embed=(id==0))
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            # TODO(brendan): .mean() for entropy?
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(1, utils.get_variable(action, requires_grad=False))

            self.entropies.append(entropy.view(-1))
            self.log_probs.append(selected_log_prob[:, 0].view(-1))

            mode = id % 2
            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:id]),
                requires_grad=False)

            if mode == 0:
                policy_id_list.append(action[:, 0])
            elif mode == 1:
                magnitude_id_list.append(action[:, 0])

        subpolicy = _get_sub_policies(policy_id_list, magnitude_id_list, self.args)
        self.entropies = torch.cat(self.entropies).sum()
        self.log_probs = torch.cat(self.log_probs).sum()

        return subpolicy


if __name__ == '__main__':
    from config import *
    args = get_args()
    controller = Controller(args).cuda()
    subplocy = controller.sample()
    for sub in subplocy:
        print(sub)
    print(controller.entropies, np.mean(controller.entropies))
    print(controller.log_probs, np.mean(controller.log_probs))
