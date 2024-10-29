import logging
import os

from dataset_creator import DatasetCreator
from omegaconf import OmegaConf

from preprocess.utils import save_json_dict

if __name__ == "__main__":
    conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(conf.config)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    word_list = [word.replace(".png", "") for word in os.listdir(yaml_conf.image_dir) if word.endswith(".png")]
    logging.info(f"Word list loaded from {yaml_conf.image_dir}. Length: {len(word_list)}\n")

    with open(yaml_conf.raw_dataset, "r") as f:
        raw_dataset = [line.rstrip("\n").replace("\t", " ") for line in f]

    dataset_creator = DatasetCreator(
        raw_dataset,
        word_list,
        yaml_conf.num_processes,
        yaml_conf.n,
        yaml_conf.sampling_negative_ratio,
    )

    positive_eval_dict, negative_eval_dict = dataset_creator.get_dataset()

    save_json_dict(yaml_conf.positive_dataset_path, positive_eval_dict)
    save_json_dict(yaml_conf.negative_dataset_path, negative_eval_dict)

    dataset_creator.stats(stats_dir=yaml_conf.stats_dir)
