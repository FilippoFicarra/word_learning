import os
from functools import partial
from multiprocessing import Pool, cpu_count

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.base_logger import logger


class Surprisal:
    """
    Class to compute extrinsic surprisal for a given model result and ground truth model result

    Parameters
    ----------
    context_type : str
        Type of context to consider. Must be one of ["positive", "negative", "combined"]
    seed : int
        Seed for the model result
    summary_output_path : str
        Path to the intermediate probabilities
    save_path : str
        Path to save the extrinsic results
    dataset : str
        Dataset to consider
    """

    def __init__(self, context_type: str, seed: int, summary_output_path: str, save_path: str, dataset: str):
        self.context_type = context_type
        self.seed = seed
        self.summary_output_path = summary_output_path
        self.save_path = save_path
        self.dataset = dataset
        self.device = "cuda" if torch.cuda.is_available() else "mps"

        try:
            intermediate_output_path = f"{self.summary_output_path}/{self.dataset}/intermediate_{self.seed}.parquet"
            self.intermdiate = pd.read_parquet(intermediate_output_path)
            logger.info(f"Loaded intermediate probabilities from {intermediate_output_path}")
        except FileNotFoundError as e:
            logger.error(f"Intermediate probabilities not found at {intermediate_output_path}")
            raise e

        if not os.path.exists(f"{self.save_path}/{self.dataset}"):
            os.makedirs(f"{self.save_path}/{self.dataset}")

        self.file_output_path = f"{self.save_path}/{self.dataset}/corpus_{self.context_type}_{self.seed}.parquet"

    def process_chunk_torch_vectorized(self, chunk, num_samples=1000, torch_seed=42, device="cpu"):
        """
        Process a chunk of data using torch vectorized operations

        Args:
            chunk (list): tuple of model_df and word_step_pairs
            num_samples (int, optional): Number of samples to draw. Defaults to 1000.
            torch_seed (int, optional): Seed for torch. Defaults to 42.
            device (str, optional): Device to use. Defaults to "cpu".

        Returns:
            results (list): List of results
        """
        torch.manual_seed(torch_seed)

        model_df, word_step_pairs = chunk
        results = []

        if self.context_type == "positive":
            sentences_to_choose = range(100)
        elif self.context_type == "negative":
            sentences_to_choose = range(100, 200)
        elif self.context_type == "combined":
            sentences_to_choose = range(200)

        for word, step in word_step_pairs:
            model_word_step_df = model_df[(model_df["word"] == word) & (model_df["step"] == step)]
            model_word_step_df = model_word_step_df[model_word_step_df["sent"].isin(sentences_to_choose)]

            # Sort by sent
            model_word_step_df = model_word_step_df.sort_values(by="sent")

            if model_word_step_df.empty:
                continue

            q_w_given_c_tensor = torch.tensor(
                model_word_step_df["p_w_given_c"].values, dtype=torch.float32, device=device
            )
            sampled_indexes = torch.randint(
                0, len(q_w_given_c_tensor), (num_samples, len(q_w_given_c_tensor)), device=device
            )

            sampled_q_w_given_c = q_w_given_c_tensor[sampled_indexes]

            mean_bootstrap_surprisal = torch.mean(-torch.log2(sampled_q_w_given_c), dim=1)

            mean_surprisal = torch.mean(mean_bootstrap_surprisal).item()
            std_surprisal = torch.std(mean_bootstrap_surprisal).item()

            results.append(
                {
                    "word": word,
                    "step": step,
                    "mean_value": mean_surprisal,
                    "std_value": std_surprisal,
                }
            )

        return results

    def get_corpus_surprisal(
        self,
        num_samples: int = 1000,
    ) -> None:
        df_step_surprisal = pd.DataFrame(columns=["word", "step", "mean_value", "std_value"])
        df_step_surprisal = df_step_surprisal.astype(
            {"word": str, "step": int, "mean_value": float, "std_value": float}
        )

        df = self.intermdiate

        unique_words = df["word"].unique()
        unique_steps = df["step"].unique()

        word_step_pairs = [(word, step) for word in unique_words for step in unique_steps]

        chunk_size = len(word_step_pairs) // (cpu_count() * 5)
        if chunk_size == 0:
            chunk_size = 1
        chunks = [(df, word_step_pairs[i : i + chunk_size]) for i in range(0, len(word_step_pairs), chunk_size)]

        with Pool(cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        partial(
                            self.process_chunk_torch_vectorized,
                            num_samples=num_samples,
                            torch_seed=42,
                            device=self.device,
                        ),
                        chunks,
                    ),
                    total=len(chunks),
                    desc="Processing",
                )
            )

        flattened_results = [item for sublist in results for item in sublist]
        df_step_surprisal = pd.DataFrame(flattened_results)

        df_step_surprisal.to_parquet(self.file_output_path, index=False)


if __name__ == "__main__":
    conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(conf.config)

    seed = conf.seed
    summary_output_path = yaml_conf.summary_output_path
    save_path = yaml_conf.save_path
    dataset = conf.dataset

    context_type = conf.context_type
    assert context_type in [
        "positive",
        "negative",
        "combined",
    ], "Context type must be either positive, negative or combined"

    surprisal = Surprisal(context_type, seed, summary_output_path, save_path, dataset)
    surprisal.get_corpus_surprisal()
