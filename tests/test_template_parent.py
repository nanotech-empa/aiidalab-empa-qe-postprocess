"""Completed SCF sources keep their actual charge folder and current form state."""
import io
from types import SimpleNamespace

import ipywidgets as ipw
import pytest
from aiida import orm
from aiida.common.links import LinkType
from plumpy import ProcessState
from aiida_pseudo.data.pseudo.upf import UpfData

from empa_qe_postprocess import unfolding

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture
def source(aiida_profile_clean, aiida_localhost):
    structure = orm.StructureData(cell=[[2.5, 0, 0], [-1.25, 2.165063509, 0], [0, 0, 15]], pbc=(True, True, True))
    structure.append_atom(position=[0, 0, 7.5], symbols="B")
    structure.store()
    code = orm.InstalledCode(computer=aiida_localhost, filepath_executable="/bin/true").store()
    pseudo = UpfData(io.BytesIO(b'<UPF version="2.0.1"><PP_HEADER element="B" z_valence="3.0"/></UPF>')).store()
    parameters = orm.Dict(dict={"CONTROL": {"calculation": "scf"}, "SYSTEM": {"ecutwfc": 30, "nbnd": 2}}).store()
    calc = orm.CalcJobNode(computer=aiida_localhost, process_type="aiida.calculations:quantumespresso.pw")
    calc.base.attributes.set("process_label", "PwCalculation")
    for name, node in {"code": code, "structure": structure, "parameters": parameters, "pseudos__B": pseudo}.items():
        calc.base.links.add_incoming(node, LinkType.INPUT_CALC, name)
    calc.store()
    bands = orm.BandsData()
    bands.set_cell_from_structure(structure)
    bands.set_kpoints([[0., 0., 0.]])
    bands.set_bands([[-1., 1.]])
    for name, node in {"remote_folder": orm.RemoteData(computer=aiida_localhost, remote_path="/unused/scf"),
                       "output_band": bands, "output_parameters": orm.Dict(dict={"fermi_energy": 0.})}.items():
        node.base.links.add_incoming(calc, LinkType.CREATE, name)
        node.store()
    calc.set_process_state(ProcessState.FINISHED)
    calc.set_exit_status(0)
    return str(calc.uuid), str(code.uuid)


def close(widget):
    for child in getattr(widget, "children", ()):
        close(child)
    widget.close()


def test_real_pw_builder_uses_completed_scf_charge_without_changing_recipe(source):
    calc = orm.load_node(source[0])
    before = calc.inputs.parameters.get_dict()
    assert unfolding.find_candidate_pw_bands_calculations(calc)[0][2].uuid == calc.uuid
    kpoints = orm.KpointsData()
    kpoints.set_kpoints([[0., 0., 0.], [.5, 0., 0.]])
    builder = unfolding.clone_pw_inputs_for_unfolding(calc, kpoints)
    assert builder.parent_folder.uuid == calc.outputs.remote_folder.uuid
    assert builder.structure.uuid == calc.inputs.structure.uuid
    assert builder.parameters.get_dict() == before | {"CONTROL": {"calculation": "bands"}}
    assert calc.inputs.parameters.get_dict() == before


def test_existing_bands_parent_is_preserved_and_missing_parent_is_explicit(source, aiida_localhost):
    calc = orm.load_node(source[0])
    old = orm.RemoteData(computer=aiida_localhost, remote_path="/unused/original").store()
    from aiida.common import AttributeDict
    bands = SimpleNamespace(inputs=AttributeDict({"parent_folder": old}))
    assert unfolding.template_parent_folder(bands).uuid == old.uuid
    bands.inputs = AttributeDict({"parameters": orm.Dict(dict={"CONTROL": {"calculation": "bands"}})})
    bands.outputs = calc.outputs
    with pytest.raises(ValueError, match="no parent charge-density"):
        unfolding.template_parent_folder(bands)


def test_actual_widget_revalidates_and_passes_scf_parent_to_real_workflow_builder(source, monkeypatch):
    from aiida_nanotech_empa.workflows.qe.banduppy import QeBanduppyUnfoldingWorkChain
    monkeypatch.setattr(unfolding, "_widgets_base_code_selector", lambda *a, **k: ipw.Text())
    monkeypatch.setattr(unfolding, "QeBanduppyUnfoldingWorkChain", QeBanduppyUnfoldingWorkChain)
    monkeypatch.setattr(unfolding, "_load_code_from_widget", lambda _: orm.load_code(source[1]))
    submissions = []
    monkeypatch.setattr(unfolding, "submit", lambda builder: submissions.append(builder) or SimpleNamespace(pk=123, uuid="test-submission-only"))
    widget = unfolding.SubmissionWidget()
    try:
        calc = orm.load_node(source[0])
        widget.pk.value = calc.pk
        widget.matrix.value = "[[1,0,0],[0,1,0],[0,0,1]]"
        widget.points_source.value = "hex2d"
        widget.path.value = "G M"
        widget.validate()
        assert widget.state["template_uuid"] == calc.uuid
        widget.prepare()
        assert len(submissions) == 1
        builder = submissions[0]
        assert builder.parent_folder.uuid == calc.outputs.remote_folder.uuid
        assert builder.template_remote_folder.uuid == calc.outputs.remote_folder.uuid
        assert builder.structure.uuid == calc.inputs.structure.uuid
        assert builder.parameters.get_dict() == calc.inputs.parameters.get_dict()
        assert builder.unfolding_parameters["labels"] == ["G", "M"]
        widget.matrix.value = "[[0,0,0],[0,1,0],[0,0,1]]"
        with pytest.raises(RuntimeError, match="was not loaded"):
            widget.prepare()
        assert widget.state == {} and len(submissions) == 1
    finally:
        close(widget)


@pytest.mark.parametrize("spacing", [0, -0.4])
def test_invalid_spacing_after_valid_load_cannot_submit(source, monkeypatch, spacing):
    """An edited draft must not turn nonpositive spacing into millions of points."""
    monkeypatch.setattr(unfolding, "_widgets_base_code_selector", lambda *a, **k: ipw.Text())
    submissions = []
    monkeypatch.setattr(unfolding, "submit", lambda builder: submissions.append(builder))
    widget = unfolding.SubmissionWidget()
    try:
        calc = orm.load_node(source[0])
        before = calc.inputs.parameters.get_dict()
        widget.pk.value = calc.pk
        widget.matrix.value = "[[1,0,0],[0,1,0],[0,0,1]]"
        widget.points_source.value = "hex2d"
        widget.path.value = "G M"
        widget.kpoint_spacing.value = 0.4
        widget.validate()
        assert widget.state["template_uuid"] == calc.uuid

        widget.kpoint_spacing.value = spacing
        assert "finite and strictly positive" in widget.segment_counts.value
        with pytest.raises(RuntimeError, match="was not loaded"):
            widget.prepare()
        assert widget.state == {}
        assert submissions == []
        assert calc.inputs.parameters.get_dict() == before

        widget.kpoint_spacing.value = 0.4
        widget.validate()
        assert widget.state["template_uuid"] == calc.uuid
        assert widget.state["segment_counts"] == [5]
        assert submissions == []
    finally:
        close(widget)
