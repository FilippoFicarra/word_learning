import warnings

import pandas as pd
from aoa_extractor import AoAExtractor
from omegaconf import OmegaConf

from src.base_logger import logger
from src.interfaces import DataHandlingArguments

warnings.filterwarnings("ignore")


def extract_from_convergence(x, y, threshold, *args) -> float:
    for t in range(len(y) - len(y) // 10):
        all_within_epsilon = True

        for s in range(t, len(y)):
            for s_prime in range(t, len(y)):
                if abs(y[s] - y[s_prime]) >= threshold:
                    all_within_epsilon = False
                    break
            if not all_within_epsilon:
                break

        if all_within_epsilon:
            return x[t], y[t]

    return None, None


FUNCTION_MAP = {
    "exponential": (None, None),  # this will be replaced by the actual functions if we need to use them
    "sinusoidal": (None, None),  # this will be replaced by the actual functions if we need to use them
}

MODE_MAP = {"convergence": extract_from_convergence}


def main():
    conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(conf.config)

    for experiment in yaml_conf.keys():
        logger.info(
            f"Processing experiment {experiment}, Mode: {yaml_conf[experiment].mode}, Function: {yaml_conf[experiment].function}"
        )
        logger.info(f"Saving to {yaml_conf[experiment].save_path}")

        result_df = pd.read_parquet(yaml_conf[experiment].result_df)

        fit_func, func = FUNCTION_MAP[yaml_conf[experiment].function]
        extract_func = MODE_MAP[yaml_conf[experiment].mode]

        data_handling_args = DataHandlingArguments(**yaml_conf[experiment].data_handling_args)

        extractor = AoAExtractor(yaml_conf[experiment].thresholds, result_df, fit_func, func, extract_func, n_cpus=8)
        aoa_df = extractor.get_aoa(data_handling_args)

        aoa_df.to_parquet(yaml_conf[experiment].save_path, index=False)


if __name__ == "__main__":
    main()
