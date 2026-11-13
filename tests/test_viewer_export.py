import io
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pytest
from aiida import orm
from aiida.common.links import LinkType
from plumpy import ProcessState

from empa_qe_postprocess import unfolding

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture
def viewer(aiida_profile_clean, aiida_localhost, tmp_path, monkeypatch):
    calculation = orm.CalcJobNode(computer=aiida_localhost)
    calculation.base.attributes.set("process_label", "QeBanduppyCalculation")
    calculation.set_process_state(ProcessState.FINISHED)
    calculation.set_exit_status(0)
    calculation.store()
    buffer = io.BytesIO()
    values = np.array([[0, 0, -1, 1], [1, 0.5, 0, 0.5], [2, 1, 1, 1.0]])
    down = values.copy()
    down[:, 2] += 0.2
    np.savez(
        buffer,
        unfolded_bandstructure=values,
        unfolded_bandstructure_up=values,
        unfolded_bandstructure_dw=down,
        spin_channels=np.array(["up", "dw"]),
        primitive_kline=np.array([0, 0.5, 1.0]),
        special_labels=np.array('{"0":"G","2":"X"}'),
    )
    buffer.seek(0)
    retrieved = orm.FolderData()
    retrieved.base.repository.put_object_from_filelike(buffer, "unfolding_bands.npz")
    retrieved.base.links.add_incoming(calculation, LinkType.CREATE, "retrieved")
    retrieved.store()
    monkeypatch.setattr(unfolding, "APP_ROOT", tmp_path)
    monkeypatch.setattr(unfolding, "RESULTS_DIR", tmp_path / "results")
    widget = unfolding.ViewerWidget(pk=calculation.pk, autoload=True)
    widget.qe_overlay.value = "none"
    widget.spin_channel.value = "both"
    widget.figure_width.value = 4
    widget.figure_height.value = 3
    widget.export_dpi.value = 150
    yield widget, calculation.uuid, tmp_path
    widget.close()
    plt.close("all")


@pytest.mark.parametrize("style", ["density", "dots"])
def test_actual_viewer_exports_both_spins_with_editable_svg(viewer, monkeypatch, style):
    widget, identifier, root = viewer
    assert widget._node_uuid == identifier
    assert not any(isinstance(value, orm.Node) for value in vars(widget).values())
    before = widget.loaded["unfolded_bandstructure_up"].copy()
    widget.unfolding_plot_style.value = style
    widget.show_legend.value = False
    output = []
    monkeypatch.setattr(unfolding, "display", output.append)
    widget.plot()
    figure = output[0]
    assert figure.axes[0].get_legend() is None
    np.testing.assert_allclose(figure.get_size_inches(), [4, 3])
    assert len(figure.axes[0].collections) >= 2
    svg = next(root.rglob("unfolded_bandstructure.svg"))
    parsed = ElementTree.fromstring(svg.read_bytes())
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert parsed.findall(".//svg:text", ns)
    assert bool(parsed.findall(".//svg:image", ns)) == (style == "density")
    png = svg.with_suffix(".png")
    assert png.read_bytes().startswith(b"\x89PNG")
    assert "Download SVG" in output[1].value
    np.testing.assert_array_equal(widget.loaded["unfolded_bandstructure_up"], before)
    assert not plt.get_fignums()


def test_changing_source_invalidates_the_loaded_arrays(viewer):
    widget, _, _ = viewer
    assert widget.loaded is not None
    widget.pk.value += 1000
    assert widget.loaded is None and widget._node_uuid is None
    assert widget.plot_button.disabled


@pytest.mark.parametrize("action", ["change", "reload", "invalid_load"])
def test_source_transition_requests_immediate_frontend_output_clear(viewer, monkeypatch, action):
    """A cleared Python identity must not leave another source's live downloads."""
    widget, _, root = viewer
    calls = {"plot": [], "details": []}
    for name, output in [("plot", widget.plot_output), ("details", widget.details)]:
        monkeypatch.setattr(
            output, "clear_output",
            lambda *args, _name=name, **kwargs: calls[_name].append(kwargs.get("wait", False)),
        )
    widget.status.value = "Saved previous-source plot and download links"
    widget.reference_energy_shift.value = 0.6
    archive = root / "previous-export.svg"
    archive.write_text("previous scientific export")

    if action == "change":
        widget.pk.value += 1000
        assert widget.status.value == ""
        assert widget.loaded is None and widget._node_uuid is None
        assert widget.plot_button.disabled
    elif action == "reload":
        widget.load()
        assert "Loaded unfolding output" in widget.status.value
        assert widget.loaded is not None
    else:
        widget.pk.value = 0
        widget.load()
        assert "Load failed" in widget.status.value
        assert widget.loaded is None and widget.plot_button.disabled
    for requests in calls.values():
        # wait=True leaves stale DOM visible until a replacement display arrives.
        assert requests and not any(requests)
    assert "Saved previous-source" not in widget.status.value
    assert widget.reference_energy_shift.value == 0.6
    assert archive.read_text() == "previous scientific export"


def test_failed_plot_requests_immediate_frontend_output_clear(viewer, monkeypatch):
    widget, _, _ = viewer
    waits = []
    monkeypatch.setattr(
        widget.plot_output, "clear_output",
        lambda *args, **kwargs: waits.append(kwargs.get("wait", False)),
    )
    widget.energy_max.value = widget.energy_min.value
    with pytest.raises(ValueError, match="E max greater"):
        widget.plot()
    assert waits == [False]
    assert not plt.get_fignums()


def test_failed_plot_does_not_leave_a_figure_allocated(viewer):
    widget, _, _ = viewer
    widget.energy_max.value = widget.energy_min.value
    with pytest.raises(ValueError, match="E max greater"):
        widget.plot()
    assert not plt.get_fignums()


@pytest.mark.parametrize("prefix", ["/", "/user/carlo/"])
def test_download_links_support_hub_prefix_and_spaces(tmp_path, monkeypatch, prefix):
    root = tmp_path / "qe app"
    monkeypatch.setattr(unfolding, "APP_ROOT", root)
    monkeypatch.setenv("JUPYTERHUB_SERVICE_PREFIX", prefix)
    link = unfolding._viewer_download_links(
        [("SVG", root / "results" / "bands plot.svg")]
    )
    assert f"{prefix.rstrip('/')}/files/apps/qe%20app/results/bands%20plot.svg" in link
