import argparse

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from signature_plotter import SignaturePlotter
from summary_plotter import SummaryPlotter

from src.base_logger import logger
from src.interfaces import DataHandlingArguments, PlotArguments

MODES = ["all", "signature", "summary"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--num_processes",
        type=int,
        default=-1,
        help="Number of processes to use for multiprocessing",
    )
    parser.add_argument(
        "-m",
        "--modes",
        type=str,
        default="summary",
        help="Modes to run. Options: all, hce, minimal_pairs, surprisal",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the config file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(conf.config)

    signature_config = yaml_conf.signature
    summary_config = yaml_conf.summary
    children_config = yaml_conf.children

    args = parse_args()

    num_processes = args.num_processes
    modes = args.modes.split(",")

    logger.info(f"Modes: {modes}, Num processes: {num_processes}")

    if any([mode not in MODES for mode in modes]):
        raise ValueError(f"Modes must be one of {MODES}")

    if modes == ["all"]:
        modes = MODES

    if "signature" in modes:
        logger.info("Signature")

        plt_args = PlotArguments(**signature_config.plt_args)
        data_handling_args = DataHandlingArguments(**signature_config.data_handling_args)

        signature_plotter = SignaturePlotter(result_dir=signature_config.result_folder)
        signature_plotter.plot_curves(plt_args=plt_args, data_handling_args=data_handling_args)

    if "summary" in modes:
        logger.info("Summary")

        summary_df = pd.read_csv(f"{summary_config.result_folder}/aoa_and_predictors.csv")

        ddf = summary_df[summary_df["lexical_class"] != "other"]

        logger.info(summary_config.save_dir)

        summary_plotter = SummaryPlotter(summary_df=summary_df, save_dir=summary_config.save_dir)

        # plot aoa vs predictors
        summary_plotter.plot_aoa_vs_predictor(plt_args=PlotArguments(fig_size=(0.65, 0.5), dpi=100))

        # plot specific words
        words = ["the", "off", "water", "puzzle", "good", "orange", "go", "climb"]
        words_children = ["the", "off", "water (beverage)", "puzzle", "good", "orange (description)", "go", "climb"]

        children_df = pd.read_csv(f"{children_config.children_folder}/child_american_proportion.csv")
        children_df = children_df.set_index(["item_id", "item_definition", "category"]).stack().reset_index()
        children_df.columns = ["item_id", "word", "category", "month", "proportion"]
        children_df = children_df.drop(columns=["item_id", "category"])
        children_df = children_df[children_df["word"].isin(words_children)]
        children_df["word"] = children_df["word"].apply(lambda x: x.split(" ")[0])

        children_df["month"] = children_df["month"].apply(lambda x: int(x))
        children_df = children_df.sort_values(by=["word"])
        children_df = children_df.rename(columns={"month": "step", "proportion": "value"})
        children_df["task"] = "children"

        aoas_filtered = summary_df[["word", "child_aoa"]]
        children_df = children_df.merge(aoas_filtered, on="word", how="left")
        # rename column to aoa_x
        children_df = children_df.rename(columns={"child_aoa": "aoa_x"})
        children_df["aoa_y"] = 0.5

        lm_signatures = pd.read_parquet(f"{signature_config.result_folder}/lm_signatures.parquet")
        lm_signatures = lm_signatures[lm_signatures["word"].isin(words)]
        lm_signatures = lm_signatures.rename(columns={"step": "step", "mean_value": "value"})
        lm_signatures = lm_signatures[(lm_signatures["seed"] == "42") & (lm_signatures["dataset"] == "unified")]

        lm_signatures = lm_signatures.sort_values(by=["word"])
        lm_signatures = lm_signatures[["word", "step", "value", "task"]]

        df = lm_signatures

        df["step"] = df["step"].apply(lambda x: np.log10(x))
        df["step"] = (df["step"] - df["step"].min()) / (df["step"].max() - df["step"].min())

        df = pd.concat([df, children_df])

        for i in range(len(words)):
            df.loc[df["word"] == words[i], "style"] = i % 2

        summary_plotter.plot_specific_words(
            plt_args=PlotArguments(fig_size=(0.5, 0.5), dpi=100), word_df=df, words=words
        )
