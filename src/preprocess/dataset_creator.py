import logging
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, List

import matplotlib.pyplot as plt


class DatasetCreator:
    def __init__(self, raw_dataset, words, num_processes, n, sampling_negative_ratio):
        self.raw_dataset = raw_dataset
        self.words = words
        self.num_processes = num_processes
        self.n = n
        self.sampling_negative_ratio = sampling_negative_ratio
        self.test_words = self.words

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def get_surprisal_eval_dataset(
        self,
        raw_dataset: List[List[str]],
        word_list: List[str],
        n: int = 50,
        sampling_ratio: int = 1.0,
    ) -> Dict[str, List[int]]:
        start_time = time.time()
        eval_dict = {}
        total_sentences = len(raw_dataset)
        complete = set()

        word_count = {word: 0 for word in word_list}

        assert sampling_ratio >= 0.0 and sampling_ratio <= 1.0, "Sample should be between 0.0 and 1.0"

        num_indices = 0 if sampling_ratio != 1.0 else 1
        sentences = {}

        for i in range(total_sentences):
            sentence_words = raw_dataset[i].split()
            # we skip sentences that are too short or too long
            if len(sentence_words) < 10 or len(sentence_words) > 200:
                continue

            for word in word_list:
                if word in complete:
                    continue

                # if we have duplicate of the same sentence, we skip it
                if word in sentences:
                    if sentence_words in sentences[word]:
                        continue

                # now we add the sentence to the list of sentences
                if word not in eval_dict:
                    eval_dict[word] = []
                    sentences[word] = []

                # check the indices of the words in sentence_words that are equal to word
                indices = [i for i, x in enumerate(sentence_words) if x == word]

                if word_count[word] < n and len(indices) == num_indices:
                    if int(len(sentence_words) * sampling_ratio) > 0:
                        word_count[word] += 1
                        eval_dict[word].append(i)
                        sentences[word].append(sentence_words)

                if word_count[word] == n:
                    del sentences[word]

            if (i + 1) % 10000 == 0:
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
                print(f"Processed {i + 1}/{total_sentences} sentences [{time.time() - start_time:.2f} seconds]")

            for word in word_list:
                if word_count[word] >= n:
                    complete.add(word)
            if len(complete) == len(word_list):
                break

        keys_to_delete = []
        for word in eval_dict.keys():
            if word_count[word] < n:
                keys_to_delete.append(word)

        for word in keys_to_delete:
            del eval_dict[word]

        return eval_dict

    def process_chunk(self, args) -> Dict[str, List[int]]:
        eval_dict = self.get_surprisal_eval_dataset(**args)
        return eval_dict

    def _get_single_dataset(self, sampling_ratio) -> Dict[str, List[int]]:
        word_list_chunks = [self.test_words[i :: self.num_processes] for i in range(self.num_processes)]

        with Pool(processes=self.num_processes) as pool:
            args = [
                {
                    "raw_dataset": self.raw_dataset,
                    "word_list": word_list_chunk,
                    "n": self.n,
                    "sampling_ratio": sampling_ratio,
                }
                for word_list_chunk in word_list_chunks
            ]
            results = pool.map(self.process_chunk, args)

        eval_dict = {}
        for result in results:
            eval_dict.update(result)

        self.test_words = [word for word in eval_dict.keys()]
        return eval_dict

    def _get_positive_dataset(self) -> Dict[str, List[int]]:
        print(f"Creating positive dataset for {len(self.test_words)} words with {self.num_processes} processes")
        return self._get_single_dataset(sampling_ratio=1.0)

    def _get_negative_dataset(self) -> Dict[str, List[int]]:
        print(f"Creating negative dataset for {len(self.test_words)} words with {self.num_processes} processes")
        return self._get_single_dataset(sampling_ratio=self.sampling_negative_ratio)

    def get_dataset(self) -> Dict[str, List[int]]:
        positive_eval_dict = self._get_positive_dataset()
        negative_eval_dict = self._get_negative_dataset()

        self.positive_eval_dict = positive_eval_dict
        self.negative_eval_dict = negative_eval_dict

        return positive_eval_dict, negative_eval_dict

    def stats(self, stats_dir: str):
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

        logging.info(f"positive dataset: {len(self.positive_eval_dict)} words")
        logging.info(f"negative dataset: {len(self.negative_eval_dict)} words")

        sum_positive = []
        sum_negative = []
        for word in self.positive_eval_dict.keys():
            sum_positive.extend([len(self.raw_dataset[i].split()) for i in self.positive_eval_dict[word]])
            sum_negative.extend([len(self.raw_dataset[i].split()) for i in self.negative_eval_dict[word]])

        logging.info(f"positive dataset: {len(sum_positive)} sentences")
        logging.info(f"negative dataset: {len(sum_negative)} sentences")

        mean_positive = sum(sum_positive) / len(sum_positive)
        mean_negative = sum(sum_negative) / len(sum_negative)

        smallest_positive = min(sum_positive)
        smallest_negative = min(sum_negative)

        logging.info(f"Smallest sentence length for positive dataset: {smallest_positive}")
        logging.info(f"Smallest sentence length for negative dataset: {smallest_negative}")

        # plot the distribution of sentence lengths
        plt.hist(sum_positive, bins=40, alpha=0.5, label="positive", color="orange")
        plt.hist(sum_negative, bins=40, alpha=0.5, label="negative", color="blue")

        plt.savefig(f"{stats_dir}/sentence_length_distribution.png")

        logging.info(f"Mean sentence length for positive dataset: {mean_positive}")
        logging.info(f"Mean sentence length for negative dataset: {mean_negative}")
