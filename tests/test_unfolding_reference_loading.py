"""Load saved references and avoid carrying their alignment to another source."""

import io

import numpy as np
import pytest
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState

from empa_qe_postprocess.unfolding import ViewerWidget

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture
def saved_workflows(aiida_profile_clean):
    assert aiida_profile_clean.is_test_profile

    def create(with_reference):
        workflow = orm.WorkChainNode()
        workflow.base.attributes.set("process_label", "QeBanduppyUnfoldingWorkChain")
        workflow.store()
        stream = io.BytesIO()
        np.savez(stream, unfolded_bandstructure=np.array([[0, 0.0, -1.0, 1.0]]))
        stream.seek(0)
        retrieved = orm.FolderData()
        retrieved.base.repository.put_object_from_filelike(
            stream, "unfolding_bands.npz"
        )
        retrieved.store()
        retrieved.base.links.add_incoming(
            workflow, link_type=LinkType.RETURN, link_label="banduppy_retrieved"
        )
        if with_reference:
            bands = orm.BandsData()
            bands.set_kpoints([[0.0, 0.0, 0.0]])
            bands.set_bands([[-1.0, 1.0]], units="eV")
            bands.store()
            bands.base.links.add_incoming(
                workflow, link_type=LinkType.RETURN, link_label="reference_bands"
            )
        workflow.set_process_state(ProcessState.FINISHED)
        workflow.set_exit_status(0)
        workflow.seal()
        return workflow.pk

    return create


@pytest.mark.parametrize("with_reference", [False, True])
def test_loading_saved_reference_or_explaining_its_absence(
    saved_workflows, with_reference
):
    pk = saved_workflows(with_reference)
    viewer = ViewerWidget(pk=pk, autoload=True)
    try:
        assert not viewer.plot_button.disabled
        assert viewer.reference_bands_pk.value == (pk if with_reference else 0)
        assert viewer.qe_overlay.value == (
            "reference" if with_reference else "supercell"
        )
        assert bool(viewer.reference_note.value) is not with_reference
        if not with_reference:
            assert "No primitive reference saved" in viewer.reference_note.value
    finally:
        viewer.close()


def test_switching_source_clears_old_reference_but_reloading_preserves_manual_choice(
    saved_workflows,
):
    first, second = saved_workflows(True), saved_workflows(False)
    viewer = ViewerWidget(pk=first, autoload=True)
    try:
        viewer.reference_energy_shift.value = 0.6
        viewer.pk.value = second
        assert viewer.plot_button.disabled
        assert viewer.reference_bands_pk.value == 0
        assert viewer.reference_energy_shift.value == 0
        assert viewer.qe_overlay.value == "supercell"
        viewer.load_button.click()
        assert not viewer.plot_button.disabled
        assert viewer.reference_bands_pk.value == 0
        # An explicit reference for this source survives a Load on the same PK.
        viewer.reference_bands_pk.value = first
        viewer.reference_energy_shift.value = -0.4
        viewer.qe_overlay.value = "reference"
        viewer.load_button.click()
        assert viewer.reference_bands_pk.value == first
        assert viewer.reference_energy_shift.value == -0.4
        assert viewer.qe_overlay.value == "reference"
    finally:
        viewer.close()
