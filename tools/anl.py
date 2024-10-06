"""
This file is used to get filter the results of the attacker.
@Data: 2024-05-29
"""

import sys
sys.path.append(".")
import argparse
import json
import os
from math import ceil

import torch
import pandas as pd

from utils.misc import extract_answer, stop_reasoning


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0, help="offset for the inference")
    return parser.parse_args()

def load_data(path):
    path = os.path.join(path, "w_gen_predictions_ans_test.json")
    data = json.load(open(path))
    return data

def load_loss(path):
    path = os.path.join(path, "losses.pt")
    loss = torch.load(path)
    return loss

def get_acc(data, losses=None, pos="last", ref=None, offset=0):
    infs = data['inference']  # all the infernece results
    # org is the reference text in the current data
    if len(data['ref_text']) != 0:  # without attack
        org = data['ref_text']
    else:
        org = [d[0] for d in data['inference']] if isinstance(data['inference'][0], list) else data['inference']
    # ref corresponds to scenario w/o cot should be the reference text under scenario w/ cot, otherwise it is the same as org
    if ref is None or len(ref) == 0:
        ref = org
    label = data['labels']
    target = data['target'] if 'target' in data else None  # for targeted attack

    num_total = 0
    num_w_cot = 0
    num_wo_cot = 0
    num_targeted_success = 0
    num_correct = 0
    num_jumped = 0
    for i in range(len(ref)):
        # skip the samples that are not correctly answered under the scenario w/ cot or don't have rationale
        if extract_answer(ref[i]) != label[i] or stop_reasoning(ref[i]):
            continue
        num_total += 1
        target_flag = False
        inf = get_inf(data, i, losses, offset=offset)
        if stop_reasoning(inf):
            num_wo_cot += 1
        else:
            num_w_cot += 1
        flag = get_tf(inf, ref[i], label[i], pos, org[i])
        ans = extract_answer(inf, pos)
        # targeted attack
        if target is not None:
            # if sample is not attacked
            if target[i] is None:
                num_jumped += 1
            # if sample is successfully targeted
            elif ans == target[i]:
                target_flag = True

        num_correct += 1 if flag else 0
        num_targeted_success += 1 if target_flag else 0

    # print(f"num_total: {num_total}, num_wo_cot: {num_wo_cot}, num_w_cot: {num_w_cot}")
    success_rate = num_targeted_success / (num_total - num_jumped) if target is not None else (num_total - num_correct) / num_total
    correct_rate = num_correct / num_total

    return round(success_rate * 100, 2), round(correct_rate * 100, 2), num_total

def get_inf(data, i, losses=None, offset=0):
    if not isinstance(data["inference"][i], list):
        inf = data["inference"][i]
    elif losses is not None:
        loss = losses[i]
        loss = torch.cat((loss, torch.zeros(1)))
        if (loss == 0).nonzero(as_tuple=False)[0].item() % 10 == 0:
            n = ceil((loss == 0).nonzero(as_tuple=False)[0].item() / 10) - 1
            n = n - 1 if n >= len(data["inference"][i]) else n
        else:
            n = len(data["inference"][i]) - 1
        n = min(n, 19 - offset)
        # if n == 0:
        #     import pdb; pdb.set_trace()
        inf = data["inference"][i][n]
    else:
        inf = data["inference"][i][-1]
    return inf

def get_tf(inf, ref, label, pos, org = None):
    ans = extract_answer(inf, pos)
    tf = (
        True
        # (correctly answered or cannot be attacked) and correctly answered under the scenario w/ cot w/o attack and correctly answered under current scenario
        if (ans == label or len(inf) == 0) and extract_answer(ref) == label and (org is None or extract_answer(org) == label)
        else False
    )
    return tf

def get_subfolders_os(path):
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    return subfolders

if __name__ == "__main__":
    args = parse_args()
    root = "./experiments"
    folders = []
    # folders += ["colm_targeted"]
    # folders += ["colm_ablation/epsilon_0.00784314", "colm_ablation/epsilon_0.01568627", "colm_ablation/epsilon_0.1254902"]
    folders += ["colm_13B", "colm_13B_targeted"]
    # folders += ["new_crt", "new_crt8"]
    # folders += ['stop_flag', 'stop_flag_8_16']
    # folders += ['imagenet', 'imagenet8', 'imagenet16']
    for folder in folders:
        paths = get_subfolders_os(os.path.join(root, folder))
        paths = sorted(paths)
        results = {}
        for i, path in enumerate(paths):
            # if "A-OK" in path:
            #     continue
            # if "without" not in path:
            #     continue
            data = load_data(path)
            losses = load_loss(path)
            ref = load_data(paths[i-3])['ref_text'] if "wo_cot" in path else None
            pos = "first" if "without" in path else "last"
            # results.update({os.path.basename(path)[30:]: {}})
            # for j in range(19, -1, -1):
            success_rate, correct_rate, num_total = get_acc(data, losses=losses, pos=pos, ref=ref, offset=0)
            results[os.path.basename(path)] = {"success_rate": success_rate, "correct_rate": correct_rate, "num_total": num_total}
            #     results[os.path.basename(path)[30:]].update({20-j: correct_rate})
        df = pd.DataFrame(results).T
        print(f"Results for {folder}")
        print(df)
