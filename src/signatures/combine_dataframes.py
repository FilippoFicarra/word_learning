import argparse
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--lm_folder",
        type=str,
        default="src/signatures/results",
        help="Path to the surprisal result folder",
    )
    parser.add_argument(
        "-c",
        "--children_folder",
        type=str,
        default="src/wordbank/data",
        help="Path to the children folder",
    )
    parser.add_argument(
        "-s",
        "--lm_stats",
        type=str,
        default="src/utils/data",
        help="Path to the language model statistics",
    )
    return parser.parse_args()


def combine_dataframes(lm_folder):
    """
    Combine surprisal dataframes.

    Args:
        lm_folder (str): Path to the surprisal result folder.

    Returns:
        pd.DataFrame: Combined surprisal dataframe.
    """
    lm_results = pd.DataFrame()

    for model in os.listdir(lm_folder):
        if os.path.isdir(os.path.join(lm_folder, model)):
            for dataset in os.listdir(os.path.join(lm_folder, model)):
                if os.path.isdir(os.path.join(lm_folder, model, dataset)):
                    for file in os.listdir(os.path.join(lm_folder, model, dataset)):
                        if file.endswith(".parquet"):
                            signature = file.split("_")[0]
                            positive = file.split("_")[1]
                            seed = file.split("_")[2].split(".")[0]
                            df = pd.read_parquet(os.path.join(lm_folder, model, dataset, file))
                            df["model"] = model
                            df["dataset"] = dataset
                            df["task"] = f"{signature}_{positive}"
                            df["seed"] = seed
                            df["log_step"] = np.log10(df["step"])
                            lm_results = pd.concat([lm_results, df])

    lm_results.to_parquet(os.path.join(lm_folder, "lm_signatures.parquet"), index=False)


def combine_aoa_and_predictors(lm_folder, children_folder, lm_stats):
    if not os.path.exists(f"{lm_folder}/signatures_aoa.parquet"):
        return
    # Read and process AOA data
    aoa_df = pd.read_parquet(f"{lm_folder}/signatures_aoa.parquet")
    aoa_df = aoa_df.drop(columns=["aoa_y"])

    concreteness_df = pd.read_csv(f"{children_folder}/concreteness_data.tsv", sep="\t")
    concreteness_df = concreteness_df[["Word", "Conc.M"]].copy()
    concreteness_df.columns = ["word", "concreteness"]

    merged_df = pd.merge(aoa_df, concreteness_df, on="word", how="left")

    mean_concreteness = merged_df["concreteness"].mean(skipna=True)
    merged_df["concreteness"] = merged_df["concreteness"].fillna(mean_concreteness)

    children_summary = pd.read_csv(f"{children_folder}/merged_data.csv")

    children_summary = children_summary[["word", "aoa", "mlu", "log_frequency", "lexical_class"]]
    children_summary.columns = ["word", "child_aoa", "child_mlu", "child_log_frequency", "lexical_class"]

    merged_df = pd.merge(merged_df, children_summary, on="word", how="left")

    lm_frequency_df = pd.DataFrame()
    for dataset in aoa_df["dataset"].unique():
        df = pd.read_csv(f"{lm_stats}/token_freq_{dataset}.csv")
        lm_frequency_df = pd.concat([lm_frequency_df, df])

    merged_df = pd.merge(merged_df, lm_frequency_df, on=["word", "dataset"], how="left")

    lm_mlu_df = pd.DataFrame()
    for dataset in aoa_df["dataset"].unique():
        df = pd.read_csv(f"{lm_stats}/mean_sentence_lengths_{dataset}.csv")
        lm_mlu_df = pd.concat([lm_mlu_df, df])

    merged_df = pd.merge(merged_df, lm_mlu_df, on=["word", "dataset"], how="left")

    merged_df["n_chars"] = merged_df["word"].apply(len)
    merged_df["log_frequency"] = np.log10(merged_df["frequency"])

    merged_df = merged_df.dropna()

    merged_df.to_csv(f"{lm_folder}/aoa_and_predictors.csv", index=False)


def main():
    args = parse_args()
    pd.set_option("display.float_format", lambda x: "%.40f" % x)

    lm_folder = args.lm_folder
    children_folder = args.children_folder
    lm_stats = args.lm_stats

    combine_dataframes(lm_folder)
    combine_aoa_and_predictors(lm_folder, children_folder, lm_stats)


if __name__ == "__main__":
    main()
