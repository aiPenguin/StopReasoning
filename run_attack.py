import argparse
import json
import os

# imports modules for registration
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2
from utils.Attacker import Attacker
from utils.datasets import A_OKVQADataset, ScienceQADataset, ImageNetDataset, MMMUDataset
from utils.misc import extract_answer, make_save_dir, save_config, save_results, setup_seeds, check_settings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    parser.add_argument("--num_beams", type=int, default=1, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")

    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="experiments/attacks")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the existing directory")
    parser.add_argument("--usr_msg", type=str, default=None, help="experiment type in the save_dir")

    parser.add_argument("--dataset", type=str, default="ScienceQA", choices=["ScienceQA", "A-OKVQA", "ImageNet", "MMMU"])
    parser.add_argument("--cls", nargs="+", default=[], help="extracted subclasses used in ImageNet dataset")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--save_every", type=int, default=100)

    # attack scenario
    parser.add_argument("--scenarios", nargs="+", default=["wo_attack"])
    parser.add_argument("--switch", type=int, default=0)
    parser.add_argument("--update", type=int, default=0)
    parser.add_argument("--stop_on_update", action="store_true")
    parser.add_argument("--targeted", action="store_true")

    # PGD attack parameters
    parser.add_argument("--epsilon", type=float, default=16 / 255)
    parser.add_argument("--alpha", type=float, default=1 / 255)
    parser.add_argument("--iter", type=int, default=100)

    # attack range
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)

    parser.add_argument("--finish_all_iter", action="store_true", help="continue attack from the last checkpoint")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to the checkpoint")
    parser.add_argument("--jump", action="store_true")

    args = parser.parse_args()
    return args


def attack(infer_dataloader, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    attacker = Attacker(chat, CONV_VISION.copy(), args, args.save_dir)

    output = {
        "inference": [],
        "ref_text": [],
        "labels": [],
        "sample_ids": [],
    }
    if args.targeted:
        output["target"] = []

    losses = torch.zeros((args.end - args.start, args.iter))
    for i in tqdm(range(args.start, args.end), position=0):
        inputs, label = infer_dataloader[i]
        output["sample_ids"].append(inputs["qid"])
        inputs.update({"label": label})
        if "wo_attack" in args.scenarios:
            llm_message, *_, = attacker.wo_attack(inputs)
            attack_label = label
        else:
            llm_message, ref_text, loss, attack_label = attacker.attack(inputs, args.scenarios)
            output["ref_text"].append(ref_text)
            losses[i - args.start] = loss
        if not isinstance(llm_message, list):
            llm_message = [llm_message]

        # update results
        output["inference"].append(llm_message)
        output["labels"].append(label)
        if args.targeted:
            output["target"].append(attack_label)
        pos = "first" if "without_reasoning" in args.scenarios else "last"
        answer = extract_answer(llm_message[-1], pos)
        output["extracted_answers"].append(answer)
        
        # stop if all samples are attacked
        if len(output["sample_ids"]) == args.end - args.start or i == len(infer_dataloader) - 1:
            break
        # partially save
        if (i + 1) % args.save_every == 0:
            output["results"]["accuracy"] = sum(output["t/f"]) / output["results"]["num_attacked"] * 100
            save_results(output, args.save_dir, losses)

    output["results"]["accuracy"] = sum(output["t/f"]) / output["results"]["num_attacked"] * 100
    save_results(output, args.save_dir, losses)


def main():
    if args.dataset == "A-OKVQA":
        infer_dataset = A_OKVQADataset(args)
        infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=0)
    elif args.dataset == "ScienceQA":
        infer_dataset = ScienceQADataset(args)
    elif args.dataset == "ImageNet":
        infer_dataset = ImageNetDataset(args)
    elif args.dataset == "MMMU":
        infer_dataset = MMMUDataset(args)
    else:
        raise NotImplementedError("Dataset not supported.")

    if args.start is None:
        args.start = 0
    if args.end is None:
        args.end = len(infer_dataset)
    args.save_dir = make_save_dir(args)
    save_config(args.save_dir, args)

    attack(infer_dataloader=infer_dataset, args=args)


# ========================================
#             Model Initialization
# ========================================

conv_dict = {"pretrain_vicuna0": CONV_VISION_Vicuna0, "pretrain_llama2": CONV_VISION_LLama2}

print("Initializing Chat")
args = parse_args()
print("====Input Arguments====")
print(json.dumps(vars(args), indent=2, sort_keys=False))
cfg = Config(args)
setup_seeds(cfg)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to("cuda:{}".format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device="cuda:{}".format(args.gpu_id))
print("Initialization Finished")

if __name__ == "__main__":
    main()
