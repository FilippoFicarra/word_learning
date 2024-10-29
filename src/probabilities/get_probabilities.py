import os

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from preprocess.utils import load_json_dict
from src.base_logger import logger
from src.probabilities.probability_toolkit import (
    ProbabilityComputerFactory,
    ProbabilityDataModule,
)

MAX_STEP_DATASET = {
    "babylm": 10000,
    "unified": 40000,
    "childes": 5000,
}
DEVICE = "cuda" if torch.cuda.is_available() else "mps"

if __name__ == "__main__":
    conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(conf.config)

    seed = conf.seed
    model_base_path = conf.model_base_path

    if "llama" not in model_base_path:
        assert seed == int(os.path.basename(model_base_path).split("-")[4]), "Seed not found in model path"
        dataset_name = os.path.basename(model_base_path).split("-")[1][1:]
    else:
        dataset_name = ""

    summary_output_path = f"{yaml_conf.summary_output_path}/{dataset_name}"
    dataset_path = yaml_conf.dataset_path
    test_data = yaml_conf.test_data
    model_name = yaml_conf.model_name
    batch_size = yaml_conf.batch_size

    if not os.path.exists(summary_output_path):
        os.makedirs(summary_output_path)

    if "llama" in model_base_path:
        intermediate_output_path = f"{summary_output_path}/intermediate.parquet"
    else:
        intermediate_output_path = f"{summary_output_path}/intermediate_{seed}.parquet"

    intermediate = pd.DataFrame(
        columns=[
            "word",
            "step",
            "sent",
            "p_c",
            "p_w_given_c",
        ]
    )
    intermediate = intermediate.astype(
        {
            "word": str,
            "step": float,
            "sent": int,
            "p_c": float,
            "p_w_given_c": float,
        }
    )

    positive_dataset = load_json_dict(f"{dataset_path}/positive_dataset.json")
    negative_dataset = load_json_dict(f"{dataset_path}/negative_dataset.json")

    with open(yaml_conf.test_data, "r") as f:
        sentences = [line.rstrip("\n").replace("\t", " ") for line in f]

    logger.info(f"Loaded dataset from {dataset_path}")
    logger.info(f"Loaded test data from {test_data}")

    if os.path.exists(intermediate_output_path):
        intermediate = pd.read_parquet(intermediate_output_path)
        logger.info(f"Loaded intermediate probabilities from {intermediate_output_path}")

    else:
        if "llama" in model_base_path:
            model_paths = [model_base_path]
        else:
            logger.info(f"Using model from {model_base_path}")
            model_paths = [
                f"{model_base_path}/{checkpoint}"
                for checkpoint in os.listdir(model_base_path)
                if os.path.isdir(f"{model_base_path}/{checkpoint}")
            ]

        logger.info(f"Intermediate probabilities not found at {intermediate_output_path}")

        for model_path in tqdm(model_paths, desc="Processing models"):
            if "llama" in model_path:
                step = -1
            else:
                step = int(model_path.split("/")[-1].split("-")[-1])
                if step > MAX_STEP_DATASET[dataset_name]:
                    logger.info(f"Skipping step {step} as it is greater than max step")
                    continue

            logger.info(f"Processing model {model_path} at step {step}")

            probability_computer = ProbabilityComputerFactory.get_probability_computer(
                model_name=model_path, step=step, device=DEVICE, cache_dir=yaml_conf.huggingface_cache_dir
            )

            logger.info(f"Model: {model_base_path}, Step: {step}")
            for word in positive_dataset.keys():
                if "'" in word:
                    continue

                positive_test_dataset = [sentences[i] for i in positive_dataset[word]]
                negative_test_dataset = [sentences[i] for i in negative_dataset[word]]
                loader = ProbabilityDataModule(
                    batch_size=batch_size,
                    num_workers=0,
                    word=word,
                    positive_test_data=positive_test_dataset,
                    negative_test_data=negative_test_dataset,
                    device=DEVICE,
                )

                intermediate_prob = probability_computer.get_probability_intermediate(loader)
                intermediate = pd.concat([intermediate, intermediate_prob], ignore_index=True)

                intermediate.to_parquet(intermediate_output_path, index=False)

        logger.info(f"Intermediate probabilities saved at {intermediate_output_path}")
