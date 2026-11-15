"""Optional shared assistant bound to the existing post-processing widgets."""

SUBMISSION_FIELDS = (
    "pk", "mode", "matrix", "primitive", "run_reference", "reference_spin",
    "reference_moments", "primitive_atom_tolerance", "reference_preview",
    "points_source", "special_points", "path", "kpoint_spacing", "segment_counts",
    "template", "pw_code", "banduppy_code", "pw_resources",
    "reference_pw_resources", "banduppy_resources", "folded_diagonalization",
    "kpoint_batch_size", "computed_bands", "band_selection", "band_energy_min",
    "band_energy_max", "band_margin", "band_window", "folded_pk",
)
VIEWER_FIELDS = (
    "pk", "energy_min", "energy_max", "threshold", "fatfactor",
    "unfolding_plot_style", "density_smear", "density_bins", "density_contrast",
    "spin_channel", "qe_overlay", "reference_bands_pk", "reference_energy_shift",
    "reference_linewidth", "show_legend", "figure_width", "figure_height", "export_dpi",
)
PAGES = {
    "submit_qe_unfolding.ipynb": (SUBMISSION_FIELDS, {
        "load_source": "validate_button", "refresh_workflow": "refresh_button",
    }, {"submit_workflow": "prepare_button"}),
    "view_qe_unfolding.ipynb": (VIEWER_FIELDS, {
        "load_result": "load_button", "plot_bands": "plot_button",
    }, {}),
    "search.ipynb": ((), {"search": "refresh_button"}, {}),
}


def make_adapter(page, widget):
    from aiidalab_ai_assistant.guided import GuidedFormAdapter, form_controls, messages

    names, draft, execution = PAGES[page]

    def fields():
        return form_controls({name: getattr(widget, name) for name in names})[0]

    def context():
        return {
            "page": page,
            "source": {key: value for key, value in getattr(widget, "state", {}).items()
                       if key in {"root_pk", "template_uuid", "template_pk", "structure_pk"}},
            "result_uuid": getattr(widget, "_node_uuid", None),
            "result_loaded": getattr(widget, "loaded", None) is not None,
            "submitted_process_uuid": getattr(widget, "submitted_process_uuid", None),
            "messages": messages([
                getattr(widget, name) for name in ("status", "details", "reference_note", "results")
                if hasattr(widget, name)
            ]),
        }

    return GuidedFormAdapter(
        "qe-postprocess", fields, context=context,
        draft_buttons=lambda: {name: getattr(widget, attr) for name, attr in draft.items()},
        execution_buttons=lambda: {name: getattr(widget, attr) for name, attr in execution.items()},
    )


def create_panel(page, widget):
    try:
        from aiidalab_ai_assistant.forms import panel
    except ModuleNotFoundError as exc:
        if not exc.name.startswith("aiidalab_ai_assistant"):
            raise
        import ipywidgets as ipw
        return ipw.HTML("AI assistant is optional; install aiidalab-ai-assistant to enable it.")
    return panel(make_adapter(page, widget))
