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

def eval_ood_detector(args, mode_args):
    base_dir = args.base_dir
    in_dataset = args.in_dataset
    out_datasets = args.out_datasets
    method = args.method
    method_args = args.method_args
    name = args.name

    in_save_dir = os.path.join(base_dir, in_dataset, method, name)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_loader_in(args, split=('val'))
    testloaderIn, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    method_args['num_classes'] = num_classes
    model = get_model(args, num_classes, load_ckpt=True)

    t0 = time.time()

    if True:
        f1 = open(os.path.join(in_save_dir, "in_scores.txt"), 'w')
        g1 = open(os.path.join(in_save_dir, "in_labels.txt"), 'w')

    ########################################In-distribution###########################################
        print("Processing in-distribution images")
        N = len(testloaderIn.dataset)
        count = 0
        for j, data in enumerate(testloaderIn):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images.float()

            with torch.no_grad():
                logits = forward_threshold(inputs, model)

                outputs = F.softmax(logits, dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)

                for k in range(preds.shape[0]):
                    g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

            scores = get_score(inputs, model, forward_threshold, method, method_args, logits=logits)
            for score in scores:
                f1.write("{}\n".format(score))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()

        f1.close()
        g1.close()

    # OOD evaluation
    for out_dataset in out_datasets:

        out_save_dir = os.path.join(in_save_dir, out_dataset)

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        testloaderOut = get_loader_out(args, (None, out_dataset), split='val').val_ood_loader
    ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution images")

        N = len(testloaderOut.dataset)
        count = 0
        for j, data in enumerate(testloaderOut):

            images, labels = data
            images = images.cuda()
            curr_batch_size = images.shape[0]

            inputs = images.float()

            with torch.no_grad():
                logits = forward_threshold(inputs, model)

            scores = get_score(inputs, model, forward_threshold, method, method_args, logits=logits)
            for score in scores:
                f2.write("{}\n".format(score))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()

        f2.close()

    return

if __name__ == '__main__':
    args.method_args = dict()
    mode_args = dict()

    if args.method == "odin":
        args.method_args['temperature'] = 1000.0
        param_dict = {
            "CIFAR-10": {
                "resnet18": 0.01,
                "resnet18_cl1.0": 0.07,
            },
            "CIFAR-100": {
                "resnet18": 0.04,
                "resnet18_cl1.0": 0.04,
            },
            "imagenet":{
                "resnet50": 0.005,
                "resnet50_cl1.0": 0.0,
                "mobilenet": 0.03,
                "mobilenet_cl1.3": 0.04,
            }
        }
        args.method_args['magnitude'] = param_dict[args.in_dataset][args.name]
    if args.method == 'mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name, 'results.npy'), allow_pickle=True)
        regressor = LogisticRegressionCV(cv=2).fit([[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]], [0,0,1,1])
        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias
        args.method_args['sample_mean'] = sample_mean
        args.method_args['precision'] = precision
        args.method_args['magnitude'] = magnitude
        args.method_args['regressor'] = regressor
        args.method_args['num_output'] = 1

    eval_ood_detector(args, mode_args)
    compute_traditional_ood(args.base_dir, args.in_dataset, args.out_datasets, args.method, args.name)
    compute_in(args.base_dir, args.in_dataset, args.method, args.name)
