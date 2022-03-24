from __future__ import print_function
import argparse
import os

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import time
from util.metrics import compute_traditional_ood, compute_in
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model
from score import get_score


def forward_fun(args):
    def forward_threshold(inputs, model):
        if args.model_arch in {'mobilenet'} :
            logits = model.forward(inputs, threshold=args.threshold)
        elif args.model_arch.find('resnet') > -1:
            logits = model.forward_threshold(inputs, threshold=args.threshold)
        else:
            logits = model(inputs)
        return logits
    return forward_threshold

args = get_args()
forward_threshold = forward_fun(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def eval_ood_detector(args):
    base_dir = args.base_dir
    in_dataset = args.in_dataset
    method = args.method
    # method_args = args.method_args
    name = args.name

    in_save_dir = os.path.join(base_dir, in_dataset, method, name)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_loader_in(args, split=('val'))
    loader, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    # method_args['num_classes'] = num_classes
    model = get_model(args, num_classes, load_ckpt=True)


    activation_log = []

    model.eval()

    count = 0
    lim = 2000
    for j, data in enumerate(loader):
        if count > lim:
            break
        images, labels = data
        images = images.cuda()
        # labels = labels.cuda()
        curr_batch_size = images.shape[0]

        inputs = images.float()

        with torch.no_grad():
            hooker_handles = []
            layer_remark = 'avgpool'
            if args.model_arch in {'densenet', 'mobilenet', 'inception'}:
                hooker_handles.append(model.avgpool.register_forward_hook(get_activation(layer_remark)))
            elif args.model_arch.find('resnet') > -1:
                hooker_handles.append(model.avgpool.register_forward_hook(get_activation(layer_remark)))

            model(inputs)
            [h.remove() for h in hooker_handles]
            feature = activation[layer_remark]

            dim = feature.shape[1]
            activation_log.append(feature.data.cpu().numpy().reshape(curr_batch_size, dim, -1).mean(2))

        count += len(images)
        print("THRESHOLD ESTIMATION {:4}/{:4} images processed".format(count, len(loader.dataset)))

    activation_log = np.concatenate(activation_log, axis=0)
    # from scipy import stats
    # stats.percentileofscore(activation_log.flatten(), 3.5)
    print(f"\nTHRESHOLD at percentile {90} is:")
    print(np.percentile(activation_log.flatten(), 90))



if __name__ == '__main__':
    eval_ood_detector(args)

