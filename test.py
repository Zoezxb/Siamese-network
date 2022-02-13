# ----------------------------------测试样本集------------------------------------------ #
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from datasets import SiameseMNIST
import torch
import SiameseNet
from torch.optim import lr_scheduler
import torch.optim as optim
cuda = torch.cuda.is_available()
cuda = False
torch.backends.cudnn.enabled = False
# ------------------------------------ 测试过程 -------------------------------------- #
def test(args):
    model = args.model
    margin = 1.
    loss_fn = SiameseNet.ContrastiveLoss(margin)

    for epoch in range(0, 1):
        for batch_idx, (pos_data_pair, pos_label_pair, neg_data_pair, neg_label_pair) in enumerate(args.test_dataloader):

            sz_pos = pos_label_pair[0].size()[1]
            sz_neg = neg_label_pair[0].size()[1]

            if not type(pos_data_pair) in (tuple, list):
                pos_data_pair = (pos_data_pair,)
                neg_data_pair = (neg_data_pair,)

            if args.cuda:
                pos_data_pair = tuple(d.cuda() for d in pos_data_pair)
                neg_data_pair = tuple(d.cuda() for d in neg_data_pair)

            # 预测正样本
            loss_pos = 0
            for i in range(sz_pos):
                if args.cuda:
                    pos_data_pair_ = (pos_data_pair[0][:,i,:,:], pos_data_pair[1][:,i,:,:])
                outputs = model(*pos_data_pair_)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)

                loss_inputs = outputs

                target = torch.tensor([0])
                if args.cuda:
                    target = target.cuda()

                if target is not None:
                    target = (target,)
                    loss_inputs += target

                loss_pos += loss_fn(*loss_inputs)

            # 预测负样本
            loss_neg = 0
            for i in range(sz_neg):
                if args.cuda:
                    neg_data_pair_ = (neg_data_pair[0][:, i, :, :], neg_data_pair[1][:, i, :, :])
                outputs = model(*neg_data_pair_)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)

                loss_inputs = outputs

                target = torch.tensor([0])
                if args.cuda:
                    target = target.cuda()

                if target is not None:
                    target = (target,)
                    loss_inputs += target

                loss_neg += loss_fn(*loss_inputs)

            print()

