import numpy as np
import pytest
import ipywidgets as ipw
from matplotlib.figure import Figure

from empa_qe_postprocess.unfolding import _plot_unfolded_banduppy_density

pytest_plugins = ["aiida.tools.pytest_fixtures"]


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


@pytest.mark.parametrize("mode,text,expected", [
    ("nonmagnetic", "", {"nspin": 1}),
    ("magnetic", '{"tot_magnetization": 2}', {"nspin": 2, "tot_magnetization": 2}),
    ("magnetic", '{"starting_magnetization": {"C": 0.3}}', {"nspin": 2, "starting_magnetization": {"C": 0.3}}),
])
def test_explicit_reference_choices(mode, text, expected):
    from empa_qe_postprocess.unfolding import reference_spin_choice
    assert reference_spin_choice({"SYSTEM": {"nspin": 2}}, mode, text) == expected


def test_reference_choice_is_required_only_when_needed():
    from empa_qe_postprocess.unfolding import reference_spin_choice
    source = {"SYSTEM": {"nspin": 2}}
    with pytest.raises(ValueError, match="explicit reference_spin_parameters"):
        reference_spin_choice(source, None)
    assert reference_spin_choice(source, None, run_reference=False) is None
    assert reference_spin_choice({"SYSTEM": {}}, None) is None


@pytest.mark.parametrize("text", ["", "[]", "{}", '{"nspin": 1}', '{"tot_magnetization": NaN}'])
def test_invalid_magnetic_reference_choices(text):
    from empa_qe_postprocess.unfolding import reference_spin_choice
    with pytest.raises(ValueError):
        reference_spin_choice({"SYSTEM": {"nspin": 2}}, "magnetic", text)


def test_submission_widget_stores_uuid_and_reloads_source(aiida_profile, monkeypatch):
    from types import SimpleNamespace
    from uuid import uuid4
    from aiida import orm
    from aiida.common import AttributeDict
    from empa_qe_postprocess import unfolding

    source = SimpleNamespace(
        uuid=str(uuid4()), pk=1, label="magnetic source",
        inputs=AttributeDict({
            "structure": orm.StructureData(cell=np.eye(3) * 4),
            "parameters": orm.Dict(dict={"SYSTEM": {"nspin": 2}}), "pseudos": {},
        }),
        outputs=SimpleNamespace(remote_folder=object()),
    )
    loaded = []
    monkeypatch.setattr(unfolding, "_widgets_base_code_selector", lambda *a, **k: ipw.Text())
    monkeypatch.setattr(unfolding.orm, "load_node", lambda identifier: loaded.append(identifier) or source)
    widget = unfolding.SubmissionWidget()
    widget.state = {"template_uuid": source.uuid}
    try:
        assert widget._template_calculation() is source
        assert widget._template_calculation() is source
        assert loaded == [source.uuid, source.uuid]
        assert all(not isinstance(value, orm.Node) for value in widget.state.values())
        widget.reference_spin.value = "nonmagnetic"
        assert widget._chosen_reference_spin(source) == {"nspin": 1}
        widget.run_reference.value = False
        assert widget._chosen_reference_spin(source) is None
        assert widget.reference_spin.disabled
    finally:
        seen = set()
        def close(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            for child in getattr(value, "children", ()):
                close(child)
            value.close()
        close(widget)
