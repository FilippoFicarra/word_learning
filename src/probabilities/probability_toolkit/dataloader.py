from typing import Any, Dict, List, Optional

import lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset


class ProbabilityDataset(Dataset):
    """
    Dataset class for surprisal task.

    Args:
        word_ids (List[str]): List of word IDs.
        sentences (List[str]): List of sentences.
        tokenizer (PreTrainedTokenizer): Tokenizer object.
        max_length (int, optional): Maximum length of the input sequence. Defaults to 512.
        device (str, optional): Device to use for tensor operations. Defaults to "cuda".
        appropriate (bool, optional): Flag indicating whether to modify the sentences. Defaults to True.
    """

    def __init__(
        self,
        word: str,
        positive_sentences: List[str],
        negative_sentences: List[str],
        device: Optional[str] = "cuda",
    ):
        self.word = word
        self.positive_sentences = positive_sentences
        self.negative_sentences = negative_sentences
        self.device = device
        np.random.seed(42)

    def __len__(self):
        return len(self.positive_sentences) + len(self.negative_sentences)

    def __getitem__(self, idx):
        if idx < len(self.positive_sentences):
            sentence = self.positive_sentences[idx]
        else:
            sentence = self.negative_sentences[idx - len(self.positive_sentences)]
            sentence_split = sentence.split()
            num_words = len(sentence_split)
            random_index = np.random.randint(1, num_words - 1)
            sentence_split[random_index] = self.word
            sentence = " ".join(sentence_split)

        return {"sentence": sentence}


class ProbabilityDataModule(pl.LightningDataModule):
    """
    Data module class for surprisal task.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.
        word (str): Word to be used.
        test_data (List[str]): List of test data sentences.
        device (str, optional): Device to use for tensor operations. Defaults to "cuda".
        appropriate (bool, optional): Flag indicating whether to modify the sentences. Defaults to True.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        word: str,
        positive_test_data: List[str],
        negative_test_data: List[str] = [],
        device: Optional[str] = "cuda",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.positive_test_data = positive_test_data
        self.negative_test_data = negative_test_data
        self.word = word
        self.device = device

    def setup(self, stage=None):
        self.test_dataset = ProbabilityDataset(
            word=self.word,
            positive_sentences=self.positive_test_data,
            negative_sentences=self.negative_test_data,
            device=self.device,
        )

    def _build_collate_fn(self, batch: List[Dict[str, Any]]):
        sentences = [item["sentence"] for item in batch]
        return sentences

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._build_collate_fn,
            shuffle=False,
        )
