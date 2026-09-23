import numpy as np
from matplotlib.figure import Figure

from empa_qe_postprocess.unfolding import _plot_unfolded_banduppy_density


def test_plot_unfolded_density_with_banduppy_1():
    """Test the released BandUPpy 1 plotting API."""
    unfolded = np.asarray(
        [
            [0, 0.0, -1.0, 1.0],
            [1, 0.5, 0.0, 0.5],
            [2, 1.0, 1.0, 1.0],
        ]
    )
    figure = Figure()
    axis = figure.subplots()

    summary = _plot_unfolded_banduppy_density(
        axis,
        unfolded,
        kline=np.asarray([0.0, 0.5, 1.0]),
        fermi=0.0,
        energy_min=-2.0,
        energy_max=2.0,
        threshold=0.0,
        smear=0.05,
        n_energy=50,
        contrast=1.0,
        cmap="viridis",
        label="unfolded",
    )

    assert summary == {"points": 3}
    assert len(axis.collections) == 1
