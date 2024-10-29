from typing import Optional, Union

import pandas as pd

from tasks.probabilities.probability_toolkit.dataloader import (
    ProbabilityDataModule,
)

TEXT_MAX_LENGTH_DEFAULT = 512


class ProbabilityComputer:
    def __init__(
        self,
        model_name: str,
        step: int,
        device: Optional[Union[int, str]] = "cuda",
    ):
        self.model_name = model_name
        self._device = device

        self.step = step
        self.intermediate_prob = pd.DataFrame(
            columns=[
                "word",
                "step",
                "sent",
                "p_c",
                "p_w_given_c",
            ]
        )
        self.intermediate_prob = self.intermediate_prob.astype(
            {
                "word": str,
                "step": float,
                "sent": int,
                "p_c": float,
                "p_w_given_c": float,
            }
        )

    def get_probability_intermediate(self, dataloader: ProbabilityDataModule) -> pd.DataFrame:
        """
        Compute the intermediate probabilities of a dataset
        """
        raise NotImplementedError("Subclasses must implement get_probability_intermediate")
