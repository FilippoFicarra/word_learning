from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PlotArguments:
    """
    Arguments for the plotter

    Attributes:
        fig_size: Tuple[int]
            Size of the figure
        dpi: int
            Dots per inch
        position: List[float]
            Position of the plot
    """

    fig_size: Tuple[int] = (3, 3)
    dpi: int = 300
    position: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.7, 0.7])


@dataclass
class DataHandlingArguments:
    """
    Arguments for the data handling

    Attributes:
        log_x: bool
            Logarithmic scale on x-axis
        log_y: bool
            Logarithmic scale on y-axis
        scale_x: bool
            Scale x-axis
        scale_y: bool
            Scale y-axis
        fit: bool
            Fit the data
    """

    log_x: bool = False
    log_y: bool = False
    scale_x: bool = False
    scale_y: bool = False
    fit: bool = False
    smooth: bool = False
