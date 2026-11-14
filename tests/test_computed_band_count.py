"""Empty-band calculation size and unfolding selection are separate decisions."""

from copy import deepcopy
import io
from types import SimpleNamespace

import ipywidgets as ipw
import numpy as np
import pytest
from aiida import orm
from aiida.engine.processes.builder import ProcessBuilder
from aiida_pseudo.data.pseudo.upf import UpfData
from aiida.common.links import LinkType

from empa_qe_postprocess import unfolding

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture
def template(monkeypatch):
    parameters = {
        "CONTROL": {"calculation": "scf"},
        "SYSTEM": {"ecutwfc": 60, "ecutrho": 480},
    }
    outputs = {
        "output_band": SimpleNamespace(
            get_bands=lambda: np.array([np.arange(-16.0, 0.0)])
        ),
        "output_parameters": SimpleNamespace(get_dict=lambda: {"fermi_energy": 0.0}),
    }
    calc = SimpleNamespace(
        inputs=SimpleNamespace(parameters=SimpleNamespace(get_dict=lambda: parameters))
    )
    monkeypatch.setattr(unfolding, "outgoing_outputs", lambda _: outputs)
    return calc, parameters


def test_default_preserves_template_energy_window_and_parameters(template):
    calc, original = template
    before = deepcopy(original)
    params, window = unfolding.qe_band_request(calc, -6, 6, 2)
    assert params == original == before
    assert "nbnd" not in params["SYSTEM"]
    assert window == unfolding.estimate_qe_band_window(calc, -6, 6, 2)
    assert window["ib_end"] == 16


def test_more_empty_states_are_computed_and_all_reach_unfolding(template):
    calc, original = template
    before = deepcopy(original)
    params, window = unfolding.qe_band_request(
        calc, -6, 6, 2, computed_bands=32, selection="all"
    )
    assert params["SYSTEM"]["nbnd"] == 32
    assert params["SYSTEM"]["ecutwfc"] == 60
    assert original == before
    assert window["ib_start"] == 0
    assert window["ib_end"] == window["selected_bands"] == 32
    assert "No template-energy filter" in unfolding.format_band_window(window)
    params["SYSTEM"]["ecutwfc"] = 1
    assert original == before


def test_existing_backend_derives_eight_reference_bands_from_32(template):
    from aiida_nanotech_empa.workflows.qe.banduppy import _reference_qe_parameters

    params, _ = unfolding.qe_band_request(
        template[0], -6, 6, 2, computed_bands=32, selection="all"
    )
    original = deepcopy(params)
    reference = _reference_qe_parameters(
        params, np.diag([2, 2, 1]), calculation="bands"
    )
    scf = _reference_qe_parameters(params, np.diag([2, 2, 1]), calculation="scf")
    assert reference["SYSTEM"]["nbnd"] == 8
    assert "nbnd" not in scf["SYSTEM"]
    assert params == original


def test_cannot_estimate_uncomputed_empty_band_energies(template):
    calc, _ = template
    with pytest.raises(ValueError, match="Select All computed bands"):
        unfolding.qe_band_request(calc, -6, 6, 2, computed_bands=32)


def test_all_inherited_states_without_band_count_override(template):
    calc, original = template
    params, window = unfolding.qe_band_request(calc, 999, 1000, 2, selection="all")
    assert params == original
    assert window["ib_start"] == 0
    assert window["ib_end"] == 16


@pytest.mark.parametrize("value", [-1, 1.5, np.nan, np.inf, True])
def test_invalid_computed_counts_do_not_reach_builder(template, value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        unfolding.qe_band_request(template[0], -6, 6, 2, computed_bands=value)


def test_real_visible_form_prepares_both_builder_counts_without_submission(
    aiida_profile, aiida_localhost, monkeypatch
):
    """Inspect the actual WorkChain builder and immutable template inputs."""
    source_params = orm.Dict(
        dict={
            "CONTROL": {"calculation": "scf"},
            "SYSTEM": {"ecutwfc": 60, "ecutrho": 480},
        }
    ).store()
    structure = orm.StructureData(cell=np.eye(3) * 5)
    structure.append_atom(position=[0, 0, 0], symbols="B")
    structure.append_atom(position=[1, 1, 1], symbols="N")
    structure.store()
    source = orm.CalcJobNode(
        computer=aiida_localhost, process_type="aiida.calculations:quantumespresso.pw"
    )
    source.base.attributes.set("process_label", "PwCalculation")
    source.base.links.add_incoming(
        source_params, link_type=LinkType.INPUT_CALC, link_label="parameters"
    )
    source.base.links.add_incoming(
        structure, link_type=LinkType.INPUT_CALC, link_label="structure"
    )
    for symbol, valence in (("B", 3), ("N", 5)):
        pseudo = UpfData(
            io.BytesIO(
                f'<UPF version="2.0.1"><PP_HEADER element="{symbol}" z_valence="{valence}.0"/></UPF>'.encode()
            )
        ).store()
        source.base.links.add_incoming(
            pseudo, link_type=LinkType.INPUT_CALC, link_label=f"pseudos__{symbol}"
        )
    source.store()
    bands = orm.BandsData()
    bands.set_kpoints([[0.0, 0.0, 0.0]])
    bands.set_bands(np.array([np.arange(-16.0, 0.0)]), units="eV")
    for name, output in {
        "output_band": bands,
        "output_parameters": orm.Dict(dict={"fermi_energy": 0.0}),
        "remote_folder": orm.RemoteData(
            computer=aiida_localhost, remote_path="/not-executed"
        ),
    }.items():
        output.base.links.add_incoming(
            source, link_type=LinkType.CREATE, link_label=name
        )
        output.store()
    code = orm.InstalledCode(
        label="pw-form-only", computer=aiida_localhost, filepath_executable="/bin/true"
    ).store()
    monkeypatch.setattr(
        unfolding,
        "_widgets_base_code_selector",
        lambda *a, **k: ipw.Text(value=str(code.uuid)),
    )
    monkeypatch.setattr(
        unfolding,
        "load_root_and_candidates",
        lambda _: (source, [(2, source.pk, source, "scf", True)]),
    )
    monkeypatch.setattr(
        unfolding, "template_parent_folder", lambda _: source.outputs.remote_folder
    )
    monkeypatch.setattr(
        unfolding, "reference_preview_text", lambda *a, **k: "Reference builder only"
    )
    submitted = []
    monkeypatch.setattr(
        unfolding,
        "submit",
        lambda builder: submitted.append(builder)
        or SimpleNamespace(pk=999, uuid="form-only-no-node"),
    )
    widget = unfolding.SubmissionWidget()
    widget.pk.value = source.pk
    widget.run_reference.value = False
    widget.computed_bands.value = 32
    widget.validate()
    assert not widget.state  # Energy window cannot qualify new states.
    widget.band_selection.value = "all"
    widget.validate()
    assert widget.state["band_window"]["ib_end"] == 32
    assert widget.band_energy_min.disabled and widget.band_energy_max.disabled
    original = source_params.get_dict()
    widget.prepare()
    assert len(submitted) == 1
    builder = submitted[0]
    assert builder.parameters["SYSTEM"]["nbnd"] == 32
    assert builder.unfolding_parameters["ib_end"] == 32
    assert builder.unfolding_parameters["ib_start"] == 0
    assert builder.unfolding_parameters["band_selection"] == "all"
    assert "band_window_energy_min" not in builder.unfolding_parameters.get_dict()
    assert builder.structure.uuid == structure.uuid
    assert source_params.get_dict() == original
    assert "nbnd" not in original["SYSTEM"]
    assert isinstance(builder, ProcessBuilder)
    # An edit after Load must reach a fresh builder, not stale validated state.
    widget.computed_bands.value = 40
    widget.prepare()
    assert len(submitted) == 2
    assert submitted[1].parameters["SYSTEM"]["nbnd"] == 40
    assert submitted[1].unfolding_parameters["ib_end"] == 40
    assert builder.parameters["SYSTEM"]["nbnd"] == 32
    widget.computed_bands.value = -1
    with pytest.raises(RuntimeError, match="was not loaded"):
        widget.prepare()
    assert widget.state == {} and len(submitted) == 2
    assert source_params.get_dict() == original
    widget.close()
