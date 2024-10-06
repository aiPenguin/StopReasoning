import json
import os

from PIL import Image
from torch.utils.data import Dataset


class A_OKVQADataset(Dataset):
    """ """

    def __init__(self, args):
        """
        Initializes a Dataset class

        Args:
        """
        self.args = args
        # self.prompt_format = args.prompt_format

        train_data = json.load(open(os.path.join(args.data_root, "A-OKVQA/aokvqa_v1p0_train.json")))
        val_data = json.load(open(os.path.join(args.data_root, "A-OKVQA/aokvqa_v1p0_val.json")))
        # WARN: use val as test
        # test_data = json.load(open(os.path.join(args.data_root, "A-OKVQA/aokvqa_v1p0_test.json")))[:20]
        test_data = json.load(open(os.path.join(args.data_root, "A-OKVQA/aokvqa_v1p0_val.json")))
        # problems = train_data + val_data + test_data
        problems = test_data
        self.data = {str(i): problems[i] for i in range(len(problems))}

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.data.keys())

    def format_problem(self, problem):
        formatted_choices = ", ".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(problem["choices"]))
        formatted_problem = f"Question: {problem['question']}\nChoices: {formatted_choices}\n"

        return formatted_problem

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        problem = self.data[str(index)]

        formatted_problem = self.format_problem(problem)
        prompt = formatted_problem + self.args.prompt

        image_id = str(problem["image_id"]).zfill(12)
        img_path = os.path.join(self.args.data_root, "coco/images", f"{image_id}.jpg")
        if os.path.exists(img_path):
            image = Image.open(img_path)
            image = image.convert("RGB")
        else:
            raise Exception("image not found")

        inputs = {"text": prompt, "image": image, "qid": index, "image_id": image_id}
        return inputs, chr(65 + problem["correct_choice_idx"])


class ScienceQADataset(Dataset):
    """ """

    def __init__(self, args):
        """
        Initializes a Dataset class

        Args:
        """
        self.args = args
        # self.prompt_format = args.prompt_format

        problems = json.load(open(os.path.join(args.data_root, "scienceqa/problems.json")))
        pid_splits = json.load(open(os.path.join(args.data_root, "scienceqa/pid_splits_w_img.json")))
        self.pid = [p for p in pid_splits["test"] if problems[p]["category"] != 'State capitals']
        self.data = {p: problems[p] for p in self.pid}


    def __len__(self):
        """returns the length of dataframe"""
        return len(self.data.keys())

    def format_problem(self, problem):
        formatted_choices = ", ".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(problem["choices"]))
        formatted_problem = f"Question: {problem['question']}\nChoices: {formatted_choices}\n"

        return formatted_problem

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        problem = self.data[self.pid[index]]

        formatted_problem = self.format_problem(problem)
        prompt = formatted_problem + self.args.prompt

        img_path = os.path.join(self.args.data_root, "scienceqa/images", f"{self.pid[index]}/image.png")
        if os.path.exists(img_path):
            image = Image.open(img_path)
            image = image.convert("RGB")
        else:
            raise Exception("image not found")

        inputs = {"text": prompt, "image": image, "qid": self.pid[index], "image_id": self.pid[index]}
        return inputs, chr(65 + problem["answer"])


class ImageNetDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.cls = args.cls
        self.image_root = os.path.join(args.data_root, "ImageNet/val_2012")
        self.subset = []
        self.build_subset()

    def build_subset(self):
        if len(self.cls) == 0:
            raise Exception("No classes specified")

        map = {}
        with open(os.path.join(self.args.data_root, "ImageNet/ILSVRC/devkit/data/map_clsloc.txt"), "r") as file:
            map_clsloc = file.readlines()
        for line in map_clsloc:
            columns = line.strip().split()
            key = columns[1]  # Second column as the key
            value = columns[2]  # Third column as the value
            map[key] = value

        with open(
            os.path.join(
                self.args.data_root, "ImageNet/ILSVRC/devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt"
            ),
            "r",
        ) as file:
            content = file.readlines()
            label_ids = [int(line.strip()) for line in content]

        full = True if "all" in self.cls else False
        for i, id in enumerate(label_ids):
            if map[str(id)] in self.cls or full:
                self.subset.append([i + 1, map[str(id)]])  # image id starts with 1

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.subset)

    def format_choices(self):
        formatted_choices = ", ".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(self.cls))
        return formatted_choices

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        data = self.subset[index]
        label = chr(65 + self.cls.index(data[1]))

        formatted_choices = self.format_choices()
        prompt = f"What is the class of the image?\nChoices: {formatted_choices}\n{self.args.prompt}"

        image_id = str(data[0]).zfill(8)
        img_path = os.path.join(self.image_root, f"ILSVRC2012_val_{image_id}.JPEG")
        if os.path.exists(img_path):
            image = Image.open(img_path)
            image = image.convert("RGB")
        else:
            raise Exception("image not found")

        inputs = {"text": prompt, "image": image, "qid": str(data[0]), "image_id": image_id}
        return inputs, label

from datasets import load_dataset, concatenate_datasets
import ast


class MMMUDataset(Dataset):
    CAT_SHORT2LONG = {
        'acc': 'Accounting',
        'agri': 'Agriculture',
        'arch': 'Architecture_and_Engineering',
        'art': 'Art',
        'art_theory': 'Art_Theory',
        'bas_med': 'Basic_Medical_Science',
        'bio': 'Biology',
        'chem': 'Chemistry',
        'cli_med': 'Clinical_Medicine',
        'cs': 'Computer_Science',
        'design': 'Design',
        'diag_med': 'Diagnostics_and_Laboratory_Medicine',
        'econ': 'Economics',
        'elec': 'Electronics',
        'ep': 'Energy_and_Power',
        'fin': 'Finance',
        'geo': 'Geography',
        'his': 'History',
        'liter': 'Literature',
        'manage': 'Manage',
        'mark': 'Marketing',
        'mate': 'Materials',
        'math': 'Math',
        'mech': 'Mechanical_Engineering',
        'music': 'Music',
        'phar': 'Pharmacy',
        'phys': 'Physics',
        'psy': 'Psychology',
        'pub_health': 'Public_Health',
        'socio': 'Sociology'
    }

    def __init__(self, args):
        """
        Initializes a Dataset class

        Args:
        """
        self.args = args
        # self.prompt_format = args.prompt_format

        # problems = json.load(open(os.path.join(args.data_root, "scienceqa/problems.json")))
        # pid_splits = json.load(open(os.path.join(args.data_root, "scienceqa/pid_splits_w_img.json")))
        # self.pid = [p for p in pid_splits["test"] if problems[p]["category"] != 'State capitals']
        # self.data = {p: problems[p] for p in self.pid}
        self.data = self.load()

    def load(self):
        # run for each subject
        sub_dataset_list = []
        for subject in MMMUDataset.CAT_SHORT2LONG.values():
            sub_dataset = load_dataset("MMMU/MMMU", subject, split="validation")
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        dataset = concatenate_datasets(sub_dataset_list)
        dataset = dataset.filter(lambda x : x['image_1'] is not None and x['image_2'] is None)
        return dataset


    def __len__(self):
        """returns the length of dataframe"""
        return len(self.data)

    def format_problem(self, problem):
        choice = ast.literal_eval(problem["options"])
        formatted_choices = ", ".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choice))
        formatted_problem = f"Question: {problem['question']}\nChoices: {formatted_choices}\n"
        formatted_problem = formatted_problem.replace('<image 1>', '')

        return formatted_problem

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        problem = self.data[index]

        formatted_problem = self.format_problem(problem)
        prompt = formatted_problem + self.args.prompt

        image = problem['image_1'].convert("RGB")

        inputs = {"text": prompt, "image": image, "qid": problem['id'], "image_id": index}
        return inputs, problem["answer"]