import argparse

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--lm_folder",
        type=str,
        default="tasks/surprisal/results",
        help="Path to the result folder",
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


if __name__ == "__main__":
    args = parse_args()
    pd.set_option("display.float_format", lambda x: "%.40f" % x)

    lm_folder = args.lm_folder
    children_folder = args.children_folder
    lm_stats = args.lm_stats

    lm_aoa_df = pd.read_parquet(f"{lm_folder}/surprisal_aoa.parquet")
    words = list(lm_aoa_df["word"].unique())

    child_aoa_df = pd.read_csv(f"{children_folder}/clean_wordbank_american.tsv", sep="\t")
    child_aoa_df = child_aoa_df[
        (child_aoa_df["language"] == "English (American)") & (child_aoa_df["measure"] == "produces")
    ]

    child_aoa_df = child_aoa_df[["CleanedSingle", "aoa", "lexical_category"]].copy()
    child_aoa_df = child_aoa_df.drop_duplicates()
    child_aoa_df = child_aoa_df.groupby(["CleanedSingle", "lexical_category"], as_index=False)["aoa"].mean()
    child_aoa_df.columns = ["word", "lexical_class", "aoa"]
    child_aoa_df["n_chars"] = child_aoa_df["word"].str.len()

    # childes_df
    childes_df = pd.read_csv(
        f"{children_folder}/childes_eng-na.tsv",
        encoding="UTF-8",
        sep="\t",
        quotechar='"',
    )

    childes_df = childes_df[["word", "word_count", "mean_sent_length"]].copy()
    childes_df.columns = ["word", "word_count", "mlu"]
    childes_df["word"] = childes_df["word"].str.lower()
    childes_df = childes_df[childes_df["word"] != ""]

    total_childes_tokens = childes_df["word_count"].sum()
    childes_df["frequency"] = (childes_df["word_count"]) / total_childes_tokens
    childes_df["log_frequency"] = np.log(childes_df["frequency"])

    merged_df = pd.merge(child_aoa_df, childes_df, on="word")

    # concreteness_df
    concreteness_df = pd.read_csv(f"{children_folder}/concreteness_data.tsv", sep="\t")
    concreteness_df = concreteness_df[["Word", "Conc.M"]].copy()
    concreteness_df.columns = ["word", "concreteness"]
    # concreteness_df = concreteness_df.dropna()

    merged_df = pd.merge(merged_df, concreteness_df, on="word", how="left")
    merged_df = merged_df[merged_df["word"].isin(words)]

    mean_concreteness = merged_df["concreteness"].mean(skipna=True)
    merged_df["concreteness"] = merged_df["concreteness"].fillna(mean_concreteness)
    merged_df = merged_df.dropna()

    merged_df.to_csv(f"{children_folder}/merged_data.csv", index=False)
