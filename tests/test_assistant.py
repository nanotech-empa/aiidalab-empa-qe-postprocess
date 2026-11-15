import pytest

import ipywidgets as ipw
import numpy as np

from empa_qe_postprocess import assistant, unfolding
ToolRegistry = pytest.importorskip("aiidalab_ai_assistant.tools").ToolRegistry

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def registry(page, widget):
    adapter = assistant.make_adapter(page, widget)
    tools = ToolRegistry()
    adapter.register(tools)
    return adapter, tools


def call(tools, name, **args):
    args["expected_revision"] = tools.call("app_get_state", {})["revision"]
    result = tools.call(name, args)
    assert "error" not in result, result
    return result


def test_submission_edits_actual_inputs_and_separates_execution(aiida_profile, monkeypatch):
    monkeypatch.setattr(unfolding, "_widgets_base_code_selector", lambda *a, **k: ipw.Text())
    widget = unfolding.SubmissionWidget()
    try:
        widget.state = {"template_uuid": "source-uuid", "matrix": np.eye(3)}
        # Use a sentinel in place of dispatch: no scientific calculation is submitted.
        widget.prepare_button._click_handlers.callbacks.clear()
        dispatched = []
        widget.prepare_button.on_click(lambda _: dispatched.append(True))
        adapter, tools = registry("submit_qe_unfolding.ipynb", widget)
        state = adapter.context()
        assert state["context"]["source"] == {"template_uuid": "source-uuid"}
        widget.state = {}
        assert any(name.startswith("pw_resources.") for name in state["fields"])
        call(tools, "app_update_draft", fields={"computed_bands": 120, "band_selection": "all"})
        assert widget.computed_bands.value == 120 and widget.band_selection.value == "all"
        assert not dispatched
        assert "error" in tools.call("app_prepare_action", {
            "action": "submit_workflow", "expected_revision": adapter.context()["revision"],
        })
        call(tools, "app_review_action", action="submit_workflow")
        pending = call(tools, "app_execute_action", action="submit_workflow")
        assert pending["status"] == "awaiting_confirmation" and not dispatched
        widget.run_reference.value = False
        assert "error" in tools.approve(pending["action_id"])
        assert not dispatched
        call(tools, "app_review_action", action="submit_workflow")
        pending = call(tools, "app_execute_action", action="submit_workflow")
        tools.approve(pending["action_id"])
        assert dispatched == [True]
    finally:
        widget.close()


def test_viewer_controls_keep_manual_reference_and_report_errors(aiida_profile):
    widget = unfolding.ViewerWidget()
    try:
        adapter, tools = registry("view_qe_unfolding.ipynb", widget)
        call(tools, "app_update_draft", fields={"energy_min": -2., "energy_max": 2., "reference_bands_pk": 123})
        assert widget.energy_min.value == -2. and widget.energy_max.value == 2.
        assert widget.reference_bands_pk.value == 123
        assert "error" in tools.call("app_prepare_action", {
            "action": "plot_bands", "expected_revision": adapter.context()["revision"],
        })
        result = call(tools, "app_prepare_action", action="load_result")
        assert result["state"]["context"]["result_loaded"] is False
        assert "Load failed" in str(result["state"]["context"]["messages"])
        assert not result["state"]["execution_actions"]
    finally:
        widget.close()


def test_search_uses_real_search_button_and_exposes_results(aiida_profile):
    widget = unfolding.SearchWidget()
    try:
        adapter, tools = registry("search.ipynb", widget)
        result = call(tools, "app_prepare_action", action="search")
        assert result["state"]["context"]["messages"]
        assert not adapter.context()["fields"]
    finally:
        widget.close()
