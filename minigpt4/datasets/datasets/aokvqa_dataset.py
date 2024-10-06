import os

from PIL import Image

from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


class AOKVQADataset(CaptionDataset):

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_id = str(ann["image_id"]).zfill(12)
        image_path = os.path.join(self.vis_root, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        formatted_rationale = " ".join(ann['rationales']) if isinstance(ann['rationales'], list) else ann['rationales']
        answer = f"Rationale: {formatted_rationale} Answer: The answer is ({chr(65 + ann['correct_choice_idx'])})"
        formatted_choices = ", ".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(ann["choices"]))
        formatted_problem = f"Question: {ann['question']}\\nChoices: {formatted_choices}\\n"
        prompt = "First, generate a rationale with at least three sentences that can be used to infer the answer to the question. At last, infer the answer according to the question, the image, and the generated rationale. The answer should be in the form 'The answer is (A).'"
        formatted_problem += prompt

        return {
            "image": image,
            "text": formatted_problem,
            "answer": answer,
            "image_id": self.img_ids[ann["image_id"]],
        }


class AOKVQAwoCoTDataset(CaptionDataset):

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_id = str(ann["image_id"]).zfill(12)
        image_path = os.path.join(self.vis_root, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        answer = f"Answer: The answer is ({chr(65 + ann['correct_choice_idx'])})"
        formatted_choices = ", ".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(ann["choices"]))
        formatted_problem = f"Question: {ann['question']}\nChoices: {formatted_choices}\n"

        return {
            "image": image,
            "text": formatted_problem,
            "answer": answer,
            "image_id": self.img_ids[ann["image_id"]],
        }
