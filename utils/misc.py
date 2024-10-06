import json
import os
import random
import re

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from nltk.tokenize import sent_tokenize
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage

# from minigpt4.common.dist_utils import get_rank


def extract_answer(pred, pos="last"):
    answer = re.findall(r"([A-Z])\)", pred)
    if len(answer) == 0:
        ans = None
    elif pos == "last":
        ans = answer[-1]
    elif pos == "first":
        ans = answer[0]
    else:
        raise NotImplementedError
    return ans


def stop_reasoning(pred, key="answer"):
    sents = sent_tokenize(pred)
    try:
        stop = True if key in sents[0] and len(re.findall(r"\(", sents[0])) == 1 else False
    except:
        stop = False
    return stop


def split_rationale_answer(pred, key="answer"):
    sents = sent_tokenize(pred)
    rationale = []
    answer = []
    last = False
    for i, sent in enumerate(sents):
        if key in sent and len(re.findall(r"\(", sent)) == 1:
            answer.append(sent)
            if i == len(sents) - 1:
                last = True
        else:
            rationale.append(sent)
    return rationale, answer, last


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results


def setup_seeds(config):
    # seed = config.run_cfg.seed + get_rank()
    seed = config.run_cfg.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def make_save_dir(args):
    # save config file
    scenario = "_".join(args.scenarios)
    if args.stop_on_update:
        scenario = f"stop_flag_{scenario}"
    if args.usr_msg is not None:
        scenario = f"{args.usr_msg}_{scenario}"
    dataset = args.dataset
    if args.dataset == "ImageNet":
        classes = "_".join(args.cls)
        dataset = f"{args.dataset}_{classes}"
    save_dir = os.path.join(
        args.output_dir, f"{dataset}_attack_{scenario}_target{args.targeted}_switch{args.switch}_update{args.update}_{args.start}_{args.end}"
    )
    # Check if the directory exists
    if os.path.isdir(save_dir):
        if not args.overwrite:
            input("Directory exists. Press Enter to continue...")
    else:
        os.makedirs(save_dir)
    return save_dir


def save_results(results, save_dir, losses):
    output_prediction_file = os.path.join(save_dir, "w_gen_predictions_ans_test.json")
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(results, indent=4))
    torch.save(losses, os.path.join(save_dir, "losses.pt"))


def save_config(save_dir, args):
    output_arg_file = os.path.join(save_dir, "input_arguments.json")
    output_cfg_file = os.path.join(save_dir, "loaded_config.json")
    with open(output_arg_file, "w") as writer:
        writer.write(json.dumps(vars(args), indent=4))
    conf = OmegaConf.load(args.cfg_path)
    OmegaConf.save(config=conf, f=output_cfg_file)


def save_image(dir, image_pt, qid):
    path_image = os.path.join(dir, "perturbed_images")
    path_image_pt = os.path.join(dir, "perturbed_images_pt")
    if not os.path.exists(path_image):
        os.makedirs(path_image)
    if not os.path.exists(path_image_pt):
        os.makedirs(path_image_pt)

    torch.save(image_pt, os.path.join(path_image_pt, f"{qid}.pt"))
    convertor = ToPILImage()
    img = convertor(image_pt)
    img.save((os.path.join(path_image, f"{qid}.jpg")))


def check_settings(args, ckpt):
    ckpt_args = json.load(open(os.path.join(ckpt, "input_arguments.json")))
    # for key in vars(args).keys():
    #     if key not in ckpt_args.keys():
    #         print(f"key {key} not found in ckpt_args")
    #         continue
    #     elif vars(args)[key] != ckpt_args[key]:
    #         print(f"key {key} mismatch: {vars(args)[key]} != {ckpt_args[key]}")
    #         return False
    flag = True
    if args.epsilon != ckpt_args["epsilon"]:
        print(f"epsilon mismatch: {args.epsilon} != {ckpt_args['epsilon']}")
        flag = False
    if args.alpha != ckpt_args["alpha"]:
        print(f"alpha mismatch: {args.alpha} != {ckpt_args['alpha']}")
        flag = False
    if args.iter != ckpt_args["iter"]:
        print(f"iter mismatch: {args.iter} != {ckpt_args['iter']}")
        flag = False
    if args.scenarios != ckpt_args["scenarios"]:
        print(f"scenarios mismatch: {args.scenarios} != {ckpt_args['scenarios']}")
        flag = False
    if args.prompt != ckpt_args["prompt"]:
        print(f"prompt mismatch: {args.prompt} != {ckpt_args['prompt']}")
        flag = False
    if args.dataset != ckpt_args["dataset"]:
        print(f"dataset mismatch: {args.dataset} != {ckpt_args['dataset']}")
        flag = False
    if args.cls != ckpt_args["cls"]:
        print(f"cls mismatch: {args.cls} != {ckpt_args['cls']}")
        flag = False
    if args.cfg_path != ckpt_args["cfg_path"]:
        print(f"cfg_path mismatch: {args.cfg_path} != {ckpt_args['cfg_path']}")
        flag = False
    return flag
