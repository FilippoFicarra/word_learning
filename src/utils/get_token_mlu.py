import os
import re
import string
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from src.base_logger import logger

custom_punctuation = string.punctuation.replace("'", "")


def process_lines(lines, target_words, custom_punctuation):
    sentence_lengths = defaultdict(list)
    for line in tqdm(lines):
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", line)
        for i, sentence in enumerate(sentences):
            for word in target_words:
                if word in sentence:
                    if len(sentence_lengths[word]) == i + 1:
                        continue
                    sentence_lengths[word].append(
                        len(sentence.translate(str.maketrans("", "", custom_punctuation)).split())
                    )
    return sentence_lengths


def main(file_path, output_folder, image_folder, dataset):
    logger.info(f"Processing {dataset}...")

    target_words = [word.split(".")[0] for word in os.listdir(image_folder) if word.endswith(".png")]
    with open(file_path, "r") as f:
        num_lines = len(f.readlines())

    chunk_size = num_lines // cpu_count()

    with open(file_path, "r") as f:
        with Pool(cpu_count()) as pool:
            jobs = []
            lines_chunk = []
            for line in f:
                lines_chunk.append(line)
                if len(lines_chunk) == chunk_size:
                    jobs.append(
                        pool.apply_async(
                            process_lines,
                            (lines_chunk, target_words, custom_punctuation),
                        )
                    )
                    lines_chunk = []

            if lines_chunk:
                jobs.append(pool.apply_async(process_lines, (lines_chunk, target_words, custom_punctuation)))

            pool.close()
            pool.join()

            # Combine results
            combined_lengths = defaultdict(list)
            for job in jobs:
                result = job.get()
                for word, lengths in result.items():
                    combined_lengths[word].extend(lengths)

        mlu_dict = {}

    for word, lengths in combined_lengths.items():
        if len(lengths) == 0:
            logger.warning(f"No sentences found for {word}")
            continue
        mlu_dict[word] = sum(lengths) / len(lengths)

    df = pd.DataFrame.from_dict({"word": list(mlu_dict.keys()), "mlu": list(mlu_dict.values()), "dataset": dataset})

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df.to_csv(f"{output_folder}/mean_sentence_lengths_{dataset}.csv", index=False)
    logger.info(f"Saved mean sentence lengths to {output_folder}/mean_sentence_lengths_{dataset}.csv")


if __name__ == "__main__":
    conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(conf.config)

    for dataset in yaml_conf:
        file_path = yaml_conf[dataset].file_path
        output_folder = yaml_conf[dataset].output_folder
        image_folder = yaml_conf[dataset].image_folder

        main(file_path, output_folder, image_folder, dataset)
