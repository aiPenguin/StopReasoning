import logging
import os
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.aokvqa_dataset import AOKVQADataset, AOKVQAwoCoTDataset
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from minigpt4.datasets.datasets.laion_dataset import LaionDataset
from minigpt4.datasets.datasets.scienceqa_dataset import ScienceQADataset, ScienceQAwoCoTDataset


@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets

@registry.register_builder("aokvqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/aokvqa/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'A-OKVQA/aokvqa_v1p0_train_aug.json')],
            # ann_paths=[os.path.join(storage_path, 'A-OKVQA/aokvqa_v1p0_train.json')],
            vis_root=os.path.join(storage_path, 'coco/images'),
        )

        return datasets


@registry.register_builder("aokvqa_wo_cot")
class AOKVQAwoCoTBuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQAwoCoTDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/aokvqa/wo_cot.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'A-OKVQA/aokvqa_v1p0_train_aug.json')],
            vis_root=os.path.join(storage_path, 'coco/images'),
        )

        return datasets

@registry.register_builder("scienceqa")
class ScienceQABuilder(BaseDatasetBuilder):
    train_dataset_cls = ScienceQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/scienceqa/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'scienceqa/data_for_gpt4_finetune.json')],
            vis_root=os.path.join(storage_path, 'scienceqa/images'),
        )

        return datasets


@registry.register_builder("scienceqa_wo_cot")
class ScienceQAwoCoTBuilder(BaseDatasetBuilder):
    train_dataset_cls = ScienceQAwoCoTDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/scienceqa/wo_cot.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'scienceqa/data_for_gpt4_finetune.json')],
            vis_root=os.path.join(storage_path, 'scienceqa/images'),
        )

        return datasets
