import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

from my_model import *
from transformers import logging
from torch.autograd import grad

import os
import re
import requests

import random
from utils import *
import design_bench
import argparse
import time

parser = argparse.ArgumentParser(description="pairwise offline")
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--task', choices=['TFBind8-Exact-v0', 'Superconductor-RandomForest-v0',
                                       'HopperController-Exact-v0', 'AntMorphology-Exact-v0', 
                                       'DKittyMorphology-Exact-v0', 'CIFARNAS-Exact-v0', 'TFBind10-Exact-v0', 
                                       'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0'], type=str, default='AntMorphology-Exact-v0')
parser.add_argument('--mode', choices=['design', 'train'], type=str, default='design')
#grad descent to train proxy
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--wd', default=0.0, type=float)
#grad ascent to obtain design
parser.add_argument('--Tmax', default=200, type=int)
parser.add_argument('--ft_lr', default=1e-3, type=float)
parser.add_argument('--topk', default=128, type=int)
parser.add_argument('--interval', default=200, type=int)
parser.add_argument('--K', default=10, type=int)
parser.add_argument('--method', choices=['ensemble', 'triteach', 'simple'], type=str, default='simple')
parser.add_argument('--majority_voting', default=1, type=int)
parser.add_argument('--soft_label', default=1, type=int)
parser.add_argument('--seed1', default=1, type=int)
parser.add_argument('--seed2', default=10, type=int)
parser.add_argument('--seed3', default=100, type=int)
#
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:' + str(args.device) if use_cuda else "cpu")
def train_proxy(args):
    task = design_bench.make(args.task)
    if args.task == 'TFBind10-Exact-v0':
        args.bs = 1024
    task_y0 = task.y
    task_x, task_y, shape0 = process_data(task, args.task, task_y0)
    task_x = torch.Tensor(task_x).to(device)
    task_y = torch.Tensor(task_y).to(device)
    L = task_x.shape[0]
    indexs = torch.randperm(L)
    task_x = task_x[indexs]
    task_y = task_y[indexs]
    train_L = int(L*0.90)
    #normalize labels
    train_labels0 = task_y[0: train_L]
    valid_labels = task_y[train_L:]
    #load logits
    train_logits0 = task_x[0: train_L]
    valid_logits = task_x[train_L:]
    T = int(train_L/args.bs) + 1
    #define model
    model = SimpleMLP(task_x.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    #begin training
    best_pcc = -1
    for e in range(args.epochs):
        #adjust lr
        adjust_learning_rate(opt, args.lr, e, args.epochs)
        #random shuffle
        indexs = torch.randperm(train_L)
        train_logits = train_logits0[indexs]
        train_labels = train_labels0[indexs]
        tmp_loss = 0
        for t in range(T):
            x_batch = train_logits[t*args.bs:(t+1)*args.bs,:]#.to(device)
            y_batch = train_labels[t*args.bs:(t+1)*args.bs]#.to(device)
            pred = model(x_batch)
            loss = torch.mean(torch.pow(pred - y_batch, 2))
            tmp_loss = tmp_loss + loss.data
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            valid_preds = model(valid_logits)
        pcc = compute_pcc(valid_preds.squeeze(), valid_labels.squeeze())
        print("epoch {} training loss {} pcc {} best pcc {}".format(e, tmp_loss/T, pcc, best_pcc))
        if pcc > best_pcc:
            best_pcc = pcc
            print("epoch {} has the best loss {}".format(e, best_pcc))
            model = model.to(torch.device('cpu'))
            torch.save(model.state_dict(), "model/" + args.task + "_proxy_" + str(args.seed) +  ".pt")
            model = model.to(device)

def design_opt(args):
    task = design_bench.make(args.task)
    load_y(args.task)
    task_y0 = task.y
    task_x, task_y, shape0 = process_data(task, args.task, task_y0)
    task_x = torch.Tensor(task_x).to(device)
    task_y = torch.Tensor(task_y).to(device)

    indexs = torch.argsort(task_y.squeeze())
    index = indexs[-args.topk:]
    x_init = copy.deepcopy(task_x[index])
    #overall before evaluation unmask1
    max_score, median_score = evaluate_sample(task, x_init, args.task, shape0)
    print("Before  max {} median {}\n".format(max_score, median_score))
    for x_i in range(x_init.shape[0]):
        if args.method == 'simple' :
            proxy = SimpleMLP(task_x.shape[1]).to(device)
            proxy.load_state_dict(torch.load("model/" + args.task + "_proxy_" + str(args.seed) + ".pt", map_location=device))
        else:
            proxy1 = SimpleMLP(task_x.shape[1]).to(device)
            proxy1.load_state_dict(torch.load("model/" + args.task + "_proxy_" + str(args.seed1) + ".pt", map_location=device))
            proxy2 = SimpleMLP(task_x.shape[1]).to(device)
            proxy2.load_state_dict(torch.load("model/" + args.task + "_proxy_" + str(args.seed2) + ".pt", map_location=device))
            proxy3 = SimpleMLP(task_x.shape[1]).to(device)
            proxy3.load_state_dict(torch.load("model/" + args.task + "_proxy_" + str(args.seed3) + ".pt", map_location=device))
        # define distill data
        candidate = x_init[x_i:x_i+1]
        #unmask2
        score_before, _ = evaluate_sample(task, candidate, args.task, shape0)
        candidate.requires_grad = True
        candidate_opt = optim.Adam([candidate], lr=args.ft_lr)
        for i in range(1, args.Tmax + 1):
            if args.method == 'simple':
                loss = -proxy(candidate)
            elif args.method == 'ensemble':
                loss = -1.0/3.0*(proxy1(candidate) + proxy2(candidate) + proxy3(candidate))
            elif args.method == 'triteach':
                adjust_proxy(proxy1, proxy2, proxy3, candidate.data, x0=task_x, y0=task_y, \
                K=args.K, majority_voting = args.majority_voting, soft_label=args.soft_label)
                loss = -1.0/3.0*(proxy1(candidate) + proxy2(candidate) + proxy3(candidate))
            candidate_opt.zero_grad()
            loss.backward()
            candidate_opt.step()
            if i % args.Tmax == 0:
                score_after, _ = evaluate_sample(task, candidate.data, args.task, shape0)
                print("candidate {} score before {} score now {}".format(x_i, score_before.squeeze(), score_after.squeeze()))
        x_init[x_i] = candidate.data
    max_score, median_score = evaluate_sample(task, x_init, args.task, shape0)
    print("After  max {} median {}\n".format(max_score, median_score))
    #save the results
    results = np.load("npy/results.npy", allow_pickle=True).item()
    key = args.task + "_" + str(args.K) + "_" + str(args.seed) + "_" \
            + str(args.seed1) + "_" + \
            str(args.seed2) + "_" + str(args.seed3) + "_" + str(args.method) + "_" \
            + str(args.majority_voting) + "_" + str(args.soft_label)
    results[key] = [max_score, median_score]
    np.save("npy/results.npy", results)

if __name__ == "__main__":
    print("this is my setting", args)
    set_seed(args.seed)
    if args.mode == 'train':
        train_proxy(args)
    elif args.mode == 'design':
        if args.task in ['TFBind8-Exact-v0', 'CIFARNAS-Exact-v0', \
        'TFBind10-Exact-v0', 'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0']:
            args.ft_lr = 1e-1
            args.Tmax = 100
        design_opt(args)
