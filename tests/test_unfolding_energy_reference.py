"""Keep stored BandUPpy E-EF and the optional reference shift distinct."""

from types import SimpleNamespace

import numpy as np
import pytest

from empa_qe_postprocess import unfolding


@pytest.mark.parametrize("channel", ["none", "up", "dw"])
@pytest.mark.parametrize("fermi", [-3.0, 2.0, 0.0, np.nan])
def test_plot_normalization_copies_the_selected_spin_channel(channel, fermi):
    stored = np.array([[0, 0.0, -2.0, 1.0], [1, 1.0, -1.0, 1.0]])
    original = stored.copy()
    data = {
        "unfolded_bandstructure": np.zeros_like(stored),
        f"unfolded_bandstructure_{channel}": stored,
        "fermi_energy": np.asarray(fermi),
    }
    result = unfolding._absolute_unfolded_bandstructure_for_channel(data, channel)
    expected = original.copy()
    if np.isfinite(fermi):
        expected[:, 2] += fermi
    np.testing.assert_allclose(result, expected)
    assert not np.shares_memory(result, stored)
    np.testing.assert_array_equal(stored, original)


@pytest.mark.parametrize("style", ["dots", "density", "lines"])
@pytest.mark.parametrize("reference_shift", [0.0, 0.6, -0.6])
def test_viewer_uses_fermi_once_and_preserves_reference_shift(
    monkeypatch, tmp_path, style, reference_shift
):
    # irrep/BandUPpy already subtracts the supplied EF while reading the bands.
    # The primitive reference has its own energy zero, as for a charged system.
    stored = np.array([[0, 0.0, -2.0, 1.0], [1, 1.0, -1.0, 1.0]])
    original = stored.copy()
    reference = SimpleNamespace(
        get_bands=lambda: np.array([[-6.0], [-5.0]]),
        labels=[(0, "G"), (1, "M")],
    )
    monkeypatch.setattr(unfolding, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        unfolding.orm, "load_node", lambda _: SimpleNamespace(pk=42)
    )
    monkeypatch.setattr(
        unfolding, "_reference_bands_overlay",
        lambda *_: {"pk": 43, "bands": reference, "fermi": -4.0},
    )
    figures = []
    monkeypatch.setattr(unfolding, "display", lambda value: figures.append(value))
    plotted = []
    helper_name = {
        "density": "_plot_unfolded_banduppy_density",
        "lines": "_plot_unfolded_smoothed_density",
    }.get(style)
    if helper_name:
        helper = getattr(unfolding, helper_name)

        def capture(axis, unfolded, **kwargs):
            plotted.append(np.asarray(unfolded)[:, 2] - kwargs["fermi"])
            return helper(axis, unfolded, **kwargs)

        monkeypatch.setattr(unfolding, helper_name, capture)

    widget = unfolding.ViewerWidget()
    try:
        widget._node_uuid = "saved-unfolding-source"
        widget.loaded = {
            "unfolded_bandstructure": stored,
            "primitive_kline": np.array([0.0, 1.0]),
            "special_labels": '{"0": "G", "1": "M"}',
            "fermi_energy": np.asarray(-3.0),
        }
        widget.energy_min.value = -3.0
        widget.energy_max.value = 0.0
        widget.figure_width.value = 2.0
        widget.figure_height.value = 2.0
        widget.export_dpi.value = 150
        widget.unfolding_plot_style.value = style
        widget.qe_overlay.value = "reference"
        widget.reference_bands_pk.value = 43
        widget.reference_energy_shift.value = reference_shift
        widget.plot()
        assert "Saved unfolded band plot" in widget.status.value
        figure = next(value for value in figures if hasattr(value, "axes"))
        axis = figure.axes[0]
        if style == "dots":
            dots = next(c for c in axis.collections if c.get_label() == "unfolded")
            plotted.append(np.asarray(dots.get_offsets())[:, 1])
        np.testing.assert_allclose(plotted[0], [-2.0, -1.0])
        line = next(line for line in axis.lines if line.get_label() == "primitive reference")
        np.testing.assert_allclose(line.get_ydata(), np.array([-2.0, -1.0]) + reference_shift)
        np.testing.assert_array_equal(stored, original)
        assert widget.reference_energy_shift.value == reference_shift
        assert (tmp_path / "qe_unfolding_view_42/unfolded_bandstructure.png").is_file()
        assert (tmp_path / "qe_unfolding_view_42/unfolded_bandstructure.svg").is_file()
    finally:
        widget.close()


def test_supercell_overlay_uses_the_same_energy_zero_without_changing_arrays():
    from matplotlib.figure import Figure

    stored = np.array([[0, 0.0, -2.0, 1.0], [1, 1.0, -1.0, 1.0]])
    original = stored.copy()
    axis = Figure().subplots()
    unfolding._plot_unfolded_supercell_dots(
        axis,
        {"unfolded_bandstructure": stored, "fermi_energy": np.asarray(-3.0)},
        kline=np.array([0.0, 1.0]),
        fermi=-3.0,
        energy_min=-3.0,
        energy_max=0.0,
        channels=["none"],
    )
    assert len(axis.collections) == 1
    np.testing.assert_allclose(axis.collections[0].get_offsets()[:, 1], [-2.0, -1.0])
    np.testing.assert_array_equal(stored, original)
