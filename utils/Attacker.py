"""

"""
import random

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.misc import extract_answer, save_image, split_rationale_answer, find_sub_list, stop_reasoning


class Attacker:
    def __init__(self, chat, chat_state, args, save_dir):
        self.chat = chat
        self.chat_state = chat_state
        self.args = args
        self.dir = save_dir

        self.epsilon = args.epsilon
        self.alpha = args.alpha
        self.num_iter = args.iter
        self.targeted = args.targeted

        std = (0.26862954, 0.26130258, 0.27577711)
        mean = (0.48145466, 0.4578275, 0.40821073)
        self.mean = torch.as_tensor(mean, device=chat.device).view(-1, 1, 1)
        self.std = torch.as_tensor(std, device=chat.device).view(-1, 1, 1)

        # HACK: set key manually
        self.key = "class" if self.args.dataset == "ImageNet" else "answer"
        self.pos = "first" if "without_reasoning" in self.args.scenarios else "last"

        self.print_flag = True

    def wo_attack(self, inputs):
        with torch.no_grad():
            ref_text, ref_logits, ref_tokens = self.generate(inputs)
        return ref_text, ref_logits, ref_tokens

    def get_reference(self, inputs):
        ref_text, ref_logits, ref_tokens = self.wo_attack(inputs)
        ref_rat_logits = (
            self.extract_rationale_logits(ref_logits, ref_tokens, ref_text)
            if "rationale" in self.args.scenarios
            else None
        )
        del ref_logits, ref_tokens
        return ref_text, ref_rat_logits

    def attack(self, inputs, scenarios):
        loss_funcs = self.get_loss_funcs(scenarios)
        ref_text, ref_rat_logits = self.get_reference(inputs)
        inputs["answer"] = ref_text
        if extract_answer(ref_text, self.pos) != inputs["label"]:
            text = [""]
            label = None
            losses = torch.zeros(self.num_iter)
        elif ref_rat_logits is None and "rationale" in self.args.scenarios:
            text = [""]
            label = None
            losses = torch.zeros(self.num_iter)
        else:
            text, losses, label = self.pgd_attack(inputs, loss_funcs, ref_rat_logits)
        return text, ref_text, losses, label

    def get_org_img_pt(self, inputs):
        original_image_pt = self.chat.vis_processor(inputs["image"]).to(self.chat.device).mul(self.std).add(self.mean)
        return original_image_pt

    def generate(self, inputs):
        # gradio_reset
        if self.chat_state is not None:
            self.chat_state.messages = []

        image = inputs["image"]
        user_message = inputs["text"]

        # upload_img
        img_list = []
        llm_message = self.chat.upload_img(image, self.chat_state, img_list)

        # gradio_ask
        self.chat.ask(user_message, self.chat_state)

        # gradio_answer
        with torch.no_grad():
            output_token, output_scores = self.chat.answer_with_grad(
                conv=self.chat_state,
                img_list=img_list,
                num_beams=self.args.num_beams,
                temperature=self.args.temperature,
                max_new_tokens=300,
                max_length=2000,
            )

        output_text = self.chat.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split("###")[0]  # remove the stop sign '###'
        output_text = output_text.split("Assistant:")[-1].strip()
        self.chat_state.messages[-1][1] = output_text

        # gradio_reset
        if self.chat_state is not None:
            self.chat_state.messages = []
        if img_list is not None:
            img_list = []

        return output_text, output_scores, output_token

    def forward(self, inputs):
        if self.chat_state is not None:
            self.chat_state.messages = []
        self.chat_state.append_message(self.chat_state.roles[0], "<Img><ImageHere></Img>")
        self.chat.ask(inputs["text"], self.chat_state)
        self.chat_state.append_message(self.chat_state.roles[1], None)
        logits = self.chat.answer_with_forward(inputs, self.chat_state)[0, :-1]  # squeeze and remove the "1" at the end
        tokens = logits.argmax(-1)
        text = self.chat.model.llama_tokenizer.decode(tokens)
        return text, logits, tokens

    def jump_already_attacked(self, inputs, image_id_path):
        ref_text = self.wo_attack(inputs)[0]
        inputs["image"] = torch.load(image_id_path).to(self.chat.device)
        text = self.wo_attack(inputs)[0]
        print(f"{inputs['qid']} is already attacked. Jumped!")
        return text, ref_text, torch.zeros(self.num_iter)

    def pgd_attack(self, inputs, loss_funcs, ref_rat_logits, ckpt_path=None, losses=None):
        message_list = []
        # initialize perturbation
        org_img_pt = self.get_org_img_pt(inputs)
        adv_img_pt = org_img_pt.clone().detach()
        adv_img_pt = adv_img_pt + torch.empty_like(adv_img_pt).uniform_(-self.epsilon, self.epsilon)
        adv_img_pt = torch.clamp(adv_img_pt, min=0, max=1).detach()

        # get label id
        if self.targeted:
            while True:
                label = random.choice(["A", "B", "C", "D"])
                if label != inputs["label"]:
                    break
        else:
            label = inputs["label"]
        label_id = (
            self.chat.model.llama_tokenizer(f"({label})", return_tensors="pt", add_special_tokens=False)
            .to(self.chat.device)
            .input_ids[0, 1]
        )
        # if the ckpt_path is not None, it means that the attack should finish all iterations from the ckpt_path
        if losses is not None:
            restart_iter = (losses == 0).nonzero(as_tuple=False)[0, 0].item() - 1
        else:
            losses = torch.zeros(self.num_iter)
            restart_iter = 0

        second = 1
        stop_flag = False
        # pgd attack main loop
        for i in tqdm(range(restart_iter, self.num_iter), position=1, leave=False):
            torch.cuda.empty_cache()
            # generate answer with perturbed image
            adv_img_pt.requires_grad = True
            inputs["image"] = adv_img_pt.sub(self.mean).div(self.std)
            text, logits, tokens = self.forward(inputs)

            # stop iter if it has wrong pred or no pred
            answer = extract_answer(text, self.pos)
            if answer is not None:
                answer_id = (
                    self.chat.model.llama_tokenizer(f"({answer})", return_tensors="pt", add_special_tokens=False)
                    .to(self.chat.device)
                    .input_ids[0, 1]
                )
                try:
                    answer_pos = torch.nonzero(tokens == answer_id)[-1].item()
                except:
                    print(f"Answer {answer} found, but token not found in {text}")
                    break
            else:
                print(f"not found answer in {text}")
                break

            # calculate lose
            loss = 0
            if len(loss_funcs) == 3 and self.args.switch != 0:
                if i % self.args.switch == 0:
                    second = 2 if second == 1 else 1
                used_loss_funcs = [loss_funcs[0], loss_funcs[second]]
            else:
                used_loss_funcs = loss_funcs
            for loss_func in used_loss_funcs:
                loss += loss_func(logits=logits, ans_pos=answer_pos, labels=label_id, ref_rat_logits=ref_rat_logits)
            loss /= len(used_loss_funcs)
            losses[i] = loss.item()
            grad = torch.autograd.grad(loss, adv_img_pt, retain_graph=False, create_graph=False, allow_unused=True)[0]

            # stop attack with the last perturbed image
            if i == self.num_iter - 1:
                break
            if (answer == label and self.targeted) or (answer != label and not self.targeted):
                if not self.args.finish_all_iter:
                    stop_flag = True
                    if not self.args.stop_on_update or self.args.update == 0:
                        break

            # update reference
            if self.args.update != 0 and (i + 1) % self.args.update == 0 and i != 0:
                llm_message, ref_rat_logits = self.get_reference(inputs)
                message_list.append(llm_message)
                if (
                    "without_reasoning" in self.args.scenarios
                    and not stop_reasoning(llm_message, self.key)
                    and extract_answer(llm_message, self.pos) is not None
                ):
                    stop_flag = False
                if stop_flag:
                    break
                inputs["answer"] = llm_message
                if ref_rat_logits is None and "rationale" in self.args.scenarios:
                    break

            # Update adversarial images
            adv_img_pt = adv_img_pt.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_img_pt - org_img_pt, min=-self.epsilon, max=self.epsilon)
            adv_img_pt = torch.clamp(org_img_pt + delta, min=0, max=1).detach()
            torch.cuda.empty_cache()
        img_pt = adv_img_pt.detach().cpu()
        save_image(self.dir, img_pt, inputs["qid"])
        llm_message = self.wo_attack(inputs)[0]
        if len(message_list) == 0 or llm_message != message_list[-1]:
            message_list.append(llm_message)
        return message_list, losses, label

    def get_loss_funcs(self, scenarios):
        loss_funcs = []
        if "baseline" in scenarios:
            loss_funcs.append(self.baseline())
        if "rationale" in scenarios:
            loss_funcs.append(self.rationale())
        if "without_reasoning" in scenarios:
            loss_funcs.append(self.without_reasoning())
        if self.print_flag:
            print(f"loss_funcs: {loss_funcs}")
            self.print_flag = False
        return loss_funcs

    def baseline(self):
        ce = nn.CrossEntropyLoss()

        def loss_func(logits=None, labels=None, ans_pos=None, **kwargs):
            loss = ce(logits[ans_pos], labels)
            if self.targeted:
                return -loss
            return loss

        return loss_func

    def extract_rationale_logits(self, logits, tokens, text):
        rationale, answer, last = split_rationale_answer(text, self.key)
        rat_logits = None
        if len(answer) == 1:
            token_answer = (
                self.chat.model.llama_tokenizer(answer[-1], return_tensors="pt", add_special_tokens=False)
                .to(self.chat.device)
                .input_ids[0, 1:]
            )
            answer_pos = find_sub_list(token_answer.tolist(), tokens.tolist())
            rat_logits = logits[: answer_pos[0][0] - 1]
        return rat_logits

    def rationale(self):
        # kld loss function
        kld = nn.KLDivLoss(reduction="batchmean", log_target=True)

        def loss_func(logits=None, ref_rat_logits=None, **kwargs):
            if logits.dim() != 2:
                logits = logits.squeeze()
            if ref_rat_logits.dim() != 2:
                ref_rat_logits = ref_rat_logits.squeeze()
            rat_len = logits.size(0) if logits.size(0) < ref_rat_logits.size(0) else ref_rat_logits.size(0)

            ref_distr = ref_rat_logits[:rat_len].log_softmax(-1)
            # cast logits to probability
            rat_distr = logits[:rat_len].log_softmax(-1)
            # calculate kl-divergence loss
            kld_loss = kld(rat_distr, ref_distr)
            return kld_loss

        return loss_func

    def without_reasoning(self):
        # loss fuction
        ce = nn.CrossEntropyLoss()

        if self.args.cfg_path == 'eval_configs/minigpt4_eval.yaml':  # vicuna
            end_sym = "###"
        else:  # llama
            end_sym = "</s>"

        def loss_func(logits=None, **kwargs):
            if self.key == "answer":
                target = (
                    self.chat.model.llama_tokenizer(
                        f"The {self.key} is ().{end_sym}", return_tensors="pt", add_special_tokens=False
                    )
                    .to(logits.device)
                    .input_ids[0]
                )
                lo = torch.cat((logits[:4], logits[5:8]), dim=0)
            elif self.key == "class":
                target = (
                    self.chat.model.llama_tokenizer(
                        f"The {self.key} of the image is ().{end_sym}", return_tensors="pt", add_special_tokens=False
                    )
                    .to(logits.device)
                    .input_ids[0]
                )
                lo = torch.cat((logits[:7], logits[8:10]), dim=0)
            else:
                 raise NotImplementedError
            prone_loss = ce(lo, target)
            loss = -prone_loss
            return loss

        return loss_func
