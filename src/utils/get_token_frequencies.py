import os
import string
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from src.base_logger import logger

custom_punctuation = string.punctuation.replace("'", "")


def process_chunk(tokens):
    counter = Counter(tokens)
    return counter


def combine_counters(counters):
    combined = defaultdict(int)
    for counter in tqdm(counters, desc="Combining counters"):
        for word, count in counter.items():
            combined[word] += count
    return combined


def read_chunk(file_path):
    with open(file_path, "r") as file:
        tokens_buffer = []
        for _, line in enumerate(file):
            tokens = line.translate(str.maketrans("", "", custom_punctuation)).split()
            tokens_buffer.extend(tokens)
            if len(tokens_buffer) >= 2000:
                yield tokens_buffer
                tokens_buffer = []
        if tokens_buffer:
            yield tokens_buffer


def parallel_token_frequency(file_path):
    start_time = time.time()

    num_processes = cpu_count()
    chunk_generator = read_chunk(file_path)

    with Pool(num_processes) as pool:
        counter_list = []
        for counter in pool.imap(process_chunk, chunk_generator):
            counter_list.append(counter)

    combined_token_freq = combine_counters(counter_list)

    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
    return combined_token_freq


def main(file_path, output_folder, image_folder, dataset):
    logger.info(f"Processing {dataset}...")
    token_freq = parallel_token_frequency(file_path)
    total_tokens = sum(token_freq.values())

    df = pd.DataFrame(columns=["word", "frequency", "dataset"])
    df = df.astype({"word": str, "frequency": float, "dataset": str})

    for word in os.listdir(image_folder):
        if word.endswith(".png"):
            word = word.split(".")[0]
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "word": [word],
                            "frequency": [token_freq[word] / total_tokens if word in token_freq else 0],
                            "dataset": [dataset],
                        }
                    ),
                ]
            )

    df.to_csv(f"{output_folder}/token_freq_{dataset}.csv", index=False)
    logger.info(f"Saved token frequencies to {output_folder}/token_freq_{dataset}.csv")


if __name__ == "__main__":
    conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(conf.config)

    for dataset in yaml_conf:
        file_path = yaml_conf[dataset].file_path
        output_folder = yaml_conf[dataset].output_folder
        image_folder = yaml_conf[dataset].image_folder

        main(file_path, output_folder, image_folder, dataset)
