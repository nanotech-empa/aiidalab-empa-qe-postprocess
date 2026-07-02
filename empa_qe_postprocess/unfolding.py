from __future__ import annotations

import ast
import contextlib
from html import escape
import io
import json
import pickle
import re
from pathlib import Path

import aiidalab_widgets_base as awb
import aiidalab_widgets_empa as awe
import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import submit
from aiida.plugins import CalculationFactory, WorkflowFactory

try:
    import seekpath
except Exception:  # pragma: no cover - optional at import time
    seekpath = None

PwCalculation = CalculationFactory("quantumespresso.pw")
QeBanduppyUnfoldingWorkChain = WorkflowFactory("nanotech_empa.qe.banduppy_unfolding")

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
RESULTS_DIR = APP_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SPECIAL_KPOINT_PRESETS = {
    "fcc": {
        "G": [0.0, 0.0, 0.0],
        "X": [0.5, 0.0, 0.5],
        "W": [0.5, 0.25, 0.75],
        "K": [0.375, 0.375, 0.75],
        "L": [0.5, 0.5, 0.5],
        "U": [0.625, 0.25, 0.625],
    },
    "hex2d": {
        "G": [0.0, 0.0, 0.0],
        "M": [0.5, 0.0, 0.0],
        "K": [1.0 / 3.0, 1.0 / 3.0, 0.0],
    },
    "square2d": {
        "G": [0.0, 0.0, 0.0],
        "X": [0.5, 0.0, 0.0],
        "M": [0.5, 0.5, 0.0],
    },
    "rect2d": {
        "G": [0.0, 0.0, 0.0],
        "X": [0.5, 0.0, 0.0],
        "Y": [0.0, 0.5, 0.0],
        "S": [0.5, 0.5, 0.0],
    },
}


def descendants(node):
    try:
        return list(node.called_descendants)
    except AttributeError:
        return []


def process_label(node):
    return getattr(node, "process_label", "") or ""


def outgoing_outputs(node):
    return {
        triple.link_label: triple.node
        for triple in node.base.links.get_outgoing(link_type=LinkType.CREATE).all()
    }


def find_candidate_pw_bands_calculations(root):
    candidates = []
    for node in [root] + descendants(root):
        if process_label(node) != "PwCalculation":
            continue
        if not getattr(node, "is_finished_ok", False):
            continue
        try:
            params = node.inputs.parameters.get_dict()
        except Exception:
            params = {}
        calculation = params.get("CONTROL", {}).get("calculation", "").lower()
        outputs = outgoing_outputs(node)
        has_bands = any(value.__class__.__name__ == "BandsData" for value in outputs.values())
        if has_bands or calculation in {"bands", "nscf"}:
            score = (2 if has_bands else 0) + (1 if calculation == "bands" else 0)
            candidates.append((score, node.pk, node, calculation, has_bands))
    return sorted(candidates, reverse=True, key=lambda item: (item[0], item[1]))


def canonical_label(label):
    label = str(label).strip().replace("Γ", "G").replace("gamma", "G").replace("Gamma", "G")
    return "G" if label.upper() in {"G", "GAMMA"} else label.upper()


def parse_loose_matrix(text, *, dtype=float, default=None):
    text = (text or "").strip()
    if not text:
        if default is None:
            raise ValueError("Matrix/lattice field is empty.")
        return np.array(default, dtype=dtype)
    value = ast.literal_eval(text)
    arr = np.array(value, dtype=object)
    rows = [list(value)] if arr.ndim == 1 else [list(row) for row in value]
    base = np.array(default if default is not None else np.eye(3), dtype=float)
    for i, row in enumerate(rows[:3]):
        if row is None:
            continue
        for j, val in enumerate(row[:3]):
            if val in (None, ""):
                continue
            base[i, j] = float(val)
    return np.array(np.rint(base), dtype=int) if dtype is int else np.array(base, dtype=float)


def format_points(points):
    return "\n".join(
        f"{label}: {coords[0]:.8g} {coords[1]:.8g} {coords[2]:.8g}"
        for label, coords in sorted(points.items())
    )


def parse_special_points(text):
    points = {}
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            label, coord_text = line.split(":", 1)
        elif "=" in line:
            label, coord_text = line.split("=", 1)
        else:
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Cannot parse special k-point line: {raw_line!r}")
            label, coord_text = parts[0], " ".join(parts[1:])
        coords = [float(x) for x in re.split(r"[\s,]+", coord_text.strip().strip("[]()")) if x]
        if len(coords) == 2:
            coords.append(0.0)
        if len(coords) != 3:
            raise ValueError(f"Special k-point {label!r} needs two or three coordinates.")
        points[canonical_label(label)] = coords
    if not points:
        raise ValueError("No special k-points are defined.")
    return points


def parse_path_sequence(text, special_points):
    labels = [canonical_label(x) for x in re.split(r"[\s,;\-→]+", (text or "").strip()) if x]
    if len(labels) < 2:
        raise ValueError("Enter at least two labels, e.g. G X W K G L.")
    missing = [label for label in labels if label not in special_points]
    if missing:
        available = ", ".join(sorted(special_points))
        raise ValueError(f"Unknown k-point labels {missing}. Available labels: {available}")
    return labels, [special_points[label] for label in labels]


def candidate_summary(calc):
    params = calc.inputs.parameters.get_dict()
    calculation = params.get("CONTROL", {}).get("calculation", "")
    kpoints = getattr(calc.inputs, "kpoints", None)
    nk = "?"
    if kpoints is not None:
        try:
            nk = len(kpoints.get_kpoints())
        except Exception:
            try:
                nk = "mesh " + str(kpoints.get_kpoints_mesh()[0])
            except Exception:
                nk = "?"
    nbnd = params.get("SYSTEM", {}).get("nbnd", "?")
    return f"PK {calc.pk} | {calculation or '?'} | kpoints {nk} | nbnd {nbnd}"


def qe_spin_unfolding_settings(calc):
    params = calc.inputs.parameters.get_dict()
    system = params.get("SYSTEM", {})
    outputs = outgoing_outputs(calc)
    output_parameters = outputs.get("output_parameters")
    parsed = output_parameters.get_dict() if output_parameters is not None else {}

    nspin = int(system.get("nspin") or parsed.get("number_of_spin_components") or 1)
    noncollinear = bool(system.get("noncolin") or parsed.get("non_colinear_calculation") or False)
    spin_orbit = bool(system.get("lspinorb") or parsed.get("spin_orbit_calculation") or False)

    if noncollinear or spin_orbit:
        return {
            "mode": "spinor",
            "spinor": None,
            "spin_channels": [None],
            "description": "noncollinear/spinor QE output",
        }
    if nspin == 2:
        return {
            "mode": "collinear",
            "spinor": None,
            "spin_channels": ["up", "dw"],
            "description": "collinear spin-polarized QE output: up and down channels",
        }
    return {
        "mode": "none",
        "spinor": None,
        "spin_channels": [None],
        "description": "non-spin-polarized QE output",
    }


def primitive_segment_lengths(structure, supercell_matrix, path_coords):
    supercell_lattice = np.asarray(structure.cell, dtype=float)
    primitive_lattice = np.linalg.solve(np.asarray(supercell_matrix, dtype=float), supercell_lattice)
    reciprocal_lattice = 2.0 * np.pi * np.linalg.inv(primitive_lattice).T
    cartesian_kpoints = np.asarray(path_coords, dtype=float) @ reciprocal_lattice
    if len(cartesian_kpoints) < 2:
        return []
    return [float(np.linalg.norm(end - start)) for start, end in zip(cartesian_kpoints[:-1], cartesian_kpoints[1:])]


def points_from_spacing(segment_lengths, spacing):
    spacing = max(float(spacing), 1e-8)
    return [max(2, int(np.ceil(length / spacing)) + 1) for length in segment_lengths]


def format_segment_sampling(labels, segment_lengths, counts):
    lines = []
    for start, end, length, count in zip(labels[:-1], labels[1:], segment_lengths, counts):
        actual = length / max(count - 1, 1)
        lines.append(f"{start}-{end}: {count} points, length {length:.3f} 1/Ang, spacing {actual:.3f} 1/Ang")
    return "\n".join(lines)


def compute_supercell_from_primitive(supercell_structure, primitive_lattice):
    sc_lattice = np.array(supercell_structure.cell, dtype=float)
    primitive_lattice = np.array(primitive_lattice, dtype=float)
    fractional_matrix = sc_lattice @ np.linalg.inv(primitive_lattice)
    integer_matrix = np.rint(fractional_matrix).astype(int)
    reconstructed = integer_matrix @ primitive_lattice
    denom = max(np.linalg.norm(sc_lattice), 1e-12)
    rel_mismatch = np.linalg.norm(sc_lattice - reconstructed) / denom
    return integer_matrix, fractional_matrix, rel_mismatch


def structure_seekpath_points(structure):
    if seekpath is None:
        raise RuntimeError("seekpath is not importable in this kernel.")
    kind_to_number = {kind.name: idx + 1 for idx, kind in enumerate(structure.kinds)}
    numbers = [kind_to_number[site.kind_name] for site in structure.sites]
    scaled = [structure.get_fractional_coordinates(site.position) for site in structure.sites]
    result = seekpath.get_path((structure.cell, scaled, numbers), with_time_reversal=True)
    points = {canonical_label(label): [float(x) for x in coords] for label, coords in result.get("point_coords", {}).items()}
    return points, result


def load_root_and_candidates(pk):
    root = orm.load_node(int(pk))
    candidates = find_candidate_pw_bands_calculations(root)
    if not candidates and process_label(root) == "PwCalculation":
        candidates = [(0, root.pk, root, "", False)]
    if not candidates:
        raise RuntimeError("No finished QE PwCalculation candidate found below this PK.")
    return root, candidates


def clone_pw_inputs_for_unfolding(template_calc, kpoints_node):
    builder = PwCalculation.get_builder()
    builder.code = template_calc.inputs.code
    builder.structure = template_calc.inputs.structure
    builder.pseudos = dict(template_calc.inputs.pseudos)
    builder.kpoints = kpoints_node
    params = template_calc.inputs.parameters.get_dict()
    params.setdefault("CONTROL", {})
    params["CONTROL"]["calculation"] = "bands"
    builder.parameters = orm.Dict(dict=params)
    if "parent_folder" in template_calc.inputs:
        builder.parent_folder = template_calc.inputs.parent_folder
    for optional in ("settings", "parallelization", "hubbard_file", "vdw_table"):
        if optional in template_calc.inputs:
            setattr(builder, optional, getattr(template_calc.inputs, optional))
    try:
        builder.metadata.options = dict(template_calc.inputs.metadata.options)
    except Exception:
        pass
    builder.metadata.label = "BandUPpy folded-kpoints QE bands"
    builder.metadata.description = f"QE bands calculation on BandUPpy folded supercell k-points. Template PwCalculation PK={template_calc.pk}."
    return builder


def state_path(folded_qe_pk):
    return DATA_DIR / f"qe_unfolding_state_pk{int(folded_qe_pk)}.pkl"


def save_state(folded_qe_pk, state):
    path = state_path(folded_qe_pk)
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(state, handle)
    return path


def load_state(folded_qe_pk):
    path = state_path(folded_qe_pk)
    if not path.exists():
        raise FileNotFoundError(f"No saved unfolding state for folded QE PK {folded_qe_pk}: {path}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def qe_prefix_path_from_remote(calc):
    params = calc.inputs.parameters.get_dict()
    control = params.get("CONTROL", {})
    prefix = control.get("prefix", "aiida")
    outdir = control.get("outdir", "./out/")
    remote = calc.outputs.remote_folder
    base = Path(remote.get_remote_path())
    outdir_path = Path(outdir)
    return (outdir_path if outdir_path.is_absolute() else base / outdir_path) / prefix





def _widgets_base_code_selector(description, default_calc_job_plugin, preferred=()):
    widget = awb.ComputationalResourcesWidget(
        description=description,
        default_calc_job_plugin=default_calc_job_plugin,
        include_setup_widget=True,
    )
    for identifier in preferred:
        try:
            widget.value = orm.load_code(identifier).uuid
            break
        except Exception:
            continue
    return widget

def _load_code_from_widget(widget):
    value = widget.value
    if value is None:
        raise ValueError(f"No code selected for {widget.description!r}.")
    return orm.load_code(value)

def html_status(widget, kind, message):
    colors = {
        "ok": ("#dff3df", "#1b5e20"),
        "warn": ("#fff4cf", "#7a4b00"),
        "err": ("#fde2e2", "#8a0000"),
        "info": ("#e7f0ff", "#173b70"),
    }
    bg, fg = colors.get(kind, colors["info"])
    widget.value = f"<div style='border:1px solid #ddd; padding:8px; width:900px; background:{bg}; color:{fg};'>{message}</div>"


class SubmissionWidget(ipw.VBox):
    def __init__(self):
        self.status = ipw.HTML()
        self.details = ipw.Output(layout=ipw.Layout(border="1px solid #ddd", padding="8px", max_height="260px", overflow="auto"))
        self.details_box = ipw.Accordion(children=[self.details], selected_index=None, layout=ipw.Layout(width="920px"))
        self.details_box.set_title(0, "Details")
        self.pk = ipw.IntText(value=0, description="QE PK", layout=ipw.Layout(width="210px"))
        self.mode = ipw.ToggleButtons(options=[("S matrix", "matrix"), ("primitive lattice", "primitive")], value="matrix", description="Reference", layout=ipw.Layout(width="430px"))
        self.matrix = ipw.Text(value="[[2, 0, 0], [0, 2, 0], [0, 0, 2]]", description="S", layout=ipw.Layout(width="640px"))
        self.primitive = ipw.Textarea(value="", description="Primitive", layout=ipw.Layout(width="760px", height="70px"))
        self.points_source = ipw.Dropdown(options=[("fcc preset", "fcc"), ("hexagonal 2D preset", "hex2d"), ("square 2D preset", "square2d"), ("rectangular 2D preset", "rect2d"), ("SeeK-path from selected structure", "seekpath"), ("custom", "custom")], value="fcc", description="Special k", layout=ipw.Layout(width="360px"))
        self.special_points = ipw.Textarea(value=format_points(SPECIAL_KPOINT_PRESETS["fcc"]), description="Available", layout=ipw.Layout(width="520px", height="150px"))
        self.path = ipw.Text(value="G X W K G L", description="Path", layout=ipw.Layout(width="640px"))
        self.kpoint_spacing = ipw.FloatText(value=0.1, description="spacing", layout=ipw.Layout(width="190px"))
        self.segment_counts = ipw.Textarea(value="Load a QE calculation to estimate points per segment.", description="segments", disabled=True, layout=ipw.Layout(width="760px", height="95px"))
        self.template = ipw.Dropdown(options=[], description="Template", layout=ipw.Layout(width="820px"))
        self.pw_code = _widgets_base_code_selector("QE pw code:", "quantumespresso.pw", preferred=("pw-7.4@localhost", "pw-7.4@daint.alps_lp83"))
        self.banduppy_code = _widgets_base_code_selector("BandUPpy code:", "nanotech_empa.qe_banduppy", preferred=("banduppy-python@localhost",))
        self.pw_resources = awe.ProcessResourcesWidget()
        self.banduppy_resources = awe.ProcessResourcesWidget()
        self.pw_resources.walltime_seconds = 1800
        self.banduppy_resources.walltime_seconds = 1800
        self._set_resources_mpi(self.pw_resources, True)
        self._set_resources_mpi(self.banduppy_resources, False)
        self.validate_button = ipw.Button(description="Load QE calculation", button_style="info", icon="search", layout=ipw.Layout(width="190px"))
        self.prepare_button = ipw.Button(description="Submit workflow", button_style="info", icon="cogs", layout=ipw.Layout(width="170px"))
        self.folded_pk = ipw.IntText(value=0, description="Workflow PK", layout=ipw.Layout(width="230px"))
        self.refresh_button = ipw.Button(description="Refresh", icon="refresh", layout=ipw.Layout(width="120px"))
        self.state = {}
        self.points_source.observe(self._set_points_from_source, names="value")
        for widget in (self.kpoint_spacing, self.path, self.special_points, self.matrix, self.primitive, self.mode):
            widget.observe(self._update_segment_counts, names="value")
        self.validate_button.on_click(self.validate)
        self.prepare_button.on_click(self.prepare)
        self.refresh_button.on_click(self.refresh)
        children = [
            ipw.HTML("<h3>Band unfolding submission</h3>"),
            ipw.HBox([self.pk, self.validate_button]),
            self.template,
            ipw.HTML("<b>Codes and resources</b>"),
            self.pw_code,
            self.pw_resources,
            self.banduppy_code,
            self.banduppy_resources,
            ipw.HTML("<b>Reference cell</b>"),
            ipw.HBox([self.mode, self.matrix]),
            self.primitive,
            ipw.HTML("<b>Primitive-cell k-path</b>"),
            ipw.HBox([self.points_source, self.path]),
            self.special_points,
            ipw.HBox([self.kpoint_spacing, ipw.HTML("<span style='line-height:32px;'>target spacing in 1/Ang</span>")]),
            self.segment_counts,
            ipw.HBox([self.prepare_button, self.folded_pk, self.refresh_button]),
            self.status,
            self.details_box,
        ]
        super().__init__(children, layout=ipw.Layout(gap="8px"))

    @staticmethod
    def _set_resources_mpi(resources, value):
        if hasattr(resources, "withmpi"):
            try:
                resources.withmpi = value
            except Exception:
                pass
        for name in ("withmpi_widget", "mpi_widget"):
            widget = getattr(resources, name, None)
            if widget is not None and hasattr(widget, "value"):
                widget.value = value

    @staticmethod
    def _resource_options(resources):
        options = {
            "max_wallclock_seconds": int(resources.walltime_seconds),
            "resources": {
                "num_machines": int(resources.nodes),
                "num_mpiprocs_per_machine": int(resources.tasks_per_node),
                "num_cores_per_mpiproc": int(resources.threads_per_task),
            },
        }
        withmpi = getattr(resources, "withmpi", None)
        if withmpi is None:
            for name in ("withmpi_widget", "mpi_widget"):
                widget = getattr(resources, name, None)
                if widget is not None and hasattr(widget, "value"):
                    withmpi = widget.value
                    break
        if withmpi is not None:
            options["withmpi"] = bool(withmpi)
        return options

    def _set_points_from_source(self, *_):
        if self.points_source.value in SPECIAL_KPOINT_PRESETS:
            self.special_points.value = format_points(SPECIAL_KPOINT_PRESETS[self.points_source.value])
        elif self.points_source.value == "seekpath" and self.state.get("template") is not None:
            points, _ = structure_seekpath_points(self.state["template"].inputs.structure)
            self.special_points.value = format_points(points)
        self._update_segment_counts()

    def _current_reference_matrix(self, structure):
        sc_default = np.eye(3)
        if self.mode.value == "matrix":
            matrix = parse_loose_matrix(self.matrix.value, dtype=int, default=sc_default)
            return matrix, matrix.astype(float), None
        primitive_default = np.array(structure.cell, dtype=float)
        primitive_lattice = parse_loose_matrix(self.primitive.value, dtype=float, default=primitive_default)
        matrix, fractional_matrix, mismatch = compute_supercell_from_primitive(structure, primitive_lattice)
        return matrix, fractional_matrix, mismatch

    def _segment_sampling(self, structure, labels, coords, matrix):
        lengths = primitive_segment_lengths(structure, matrix, coords)
        counts = points_from_spacing(lengths, self.kpoint_spacing.value)
        return lengths, counts

    def _update_segment_counts(self, *_):
        template = self.state.get("template")
        if template is None:
            return
        try:
            structure = template.inputs.structure
            matrix, _, _ = self._current_reference_matrix(structure)
            special_points = parse_special_points(self.special_points.value)
            labels, coords = parse_path_sequence(self.path.value, special_points)
            lengths, counts = self._segment_sampling(structure, labels, coords, matrix)
            self.segment_counts.value = format_segment_sampling(labels, lengths, counts)
        except Exception as exc:
            self.segment_counts.value = f"Cannot estimate segment sampling yet: {type(exc).__name__}: {exc}"

    def validate(self, _=None):
        self.details.clear_output()
        try:
            root, candidates = load_root_and_candidates(self.pk.value)
            self.template.options = [(candidate_summary(item[2]), i) for i, item in enumerate(candidates)]
            if self.template.value is None:
                self.template.value = 0
            selected = candidates[self.template.value][2]
            structure = selected.inputs.structure
            matrix, fractional_matrix, mismatch = self._current_reference_matrix(structure)
            if self.mode.value == "primitive":
                self.matrix.value = json.dumps(matrix.tolist())
            if self.points_source.value != "custom":
                self._set_points_from_source()
            special_points = parse_special_points(self.special_points.value)
            labels, coords = parse_path_sequence(self.path.value, special_points)
            segment_lengths, segment_counts = self._segment_sampling(structure, labels, coords, matrix)
            self.segment_counts.value = format_segment_sampling(labels, segment_lengths, segment_counts)
            spin_settings = qe_spin_unfolding_settings(selected)
            det = round(abs(float(np.linalg.det(matrix))))
            self.state = {
                "root_pk": root.pk,
                "template": selected,
                "template_pk": selected.pk,
                "structure_pk": structure.pk,
                "matrix": matrix,
                "fractional_matrix": fractional_matrix,
                "mismatch": mismatch,
                "special_points": special_points,
                "labels": labels,
                "coords": coords,
                "segment_lengths": segment_lengths,
                "segment_counts": segment_counts,
                "spin_settings": spin_settings,
            }
            mismatch_text = "" if mismatch is None else f" | mismatch <code>{mismatch:.2e}</code>"
            html_status(self.status, "ok", f"<b>QE calculation loaded.</b> Template PK <code>{selected.pk}</code> | structure <code>{structure.get_formula()}</code> ({len(structure.sites)} atoms). The path and reference-cell inputs will be read again when submitting.")
            with self.details:
                print(f"Root: PK {root.pk} | {process_label(root)} | state {getattr(root, 'process_state', None)}")
                print(f"Template: {candidate_summary(selected)}")
                print(f"Structure: PK {structure.pk} | {structure.get_formula()} | atoms {len(structure.sites)}")
                print("Supercell matrix:")
                print(matrix)
                if mismatch is not None:
                    print("Fractional matrix before rounding:")
                    print(np.array2string(fractional_matrix, precision=5, suppress_small=True))
                    print(f"Relative lattice mismatch after rounding: {mismatch:.3e}")
                print("Available special k-points:")
                for label in sorted(special_points):
                    print(f"  {label:>6s}: {np.array2string(np.array(special_points[label]), precision=5, suppress_small=True)}")
                print("Spin mode:", spin_settings["description"])
                print(f"Target k-point spacing: {float(self.kpoint_spacing.value):.4g} 1/Ang")
                print("Points per segment:")
                print(format_segment_sampling(labels, segment_lengths, segment_counts))
        except Exception as exc:
            html_status(self.status, "err", f"<b>Validation failed:</b> {type(exc).__name__}: {exc}")
            with self.details:
                print(f"Validation failed: {type(exc).__name__}: {exc}")


    def prepare(self, _=None):
        try:
            # Always re-read the visible widget values before submission.
            # Otherwise changing the path after validation would submit stale state.
            self.validate()
            if not self.state:
                raise RuntimeError("QE calculation was not loaded; click Load QE calculation first.")
            template = self.state["template"]
            matrix = np.array(self.state["matrix"], dtype=int)
            labels = list(self.state["labels"])
            coords = [list(c) for c in self.state["coords"]]
            segment_counts = [int(x) for x in self.state["segment_counts"]]
            spin_settings = dict(self.state["spin_settings"])

            builder = QeBanduppyUnfoldingWorkChain.get_builder()
            builder.pw_code = _load_code_from_widget(self.pw_code)
            builder.banduppy_code = _load_code_from_widget(self.banduppy_code)
            builder.structure = template.inputs.structure
            builder.parameters = template.inputs.parameters
            builder.pseudos = dict(template.inputs.pseudos)
            if "parent_folder" in template.inputs:
                builder.parent_folder = template.inputs.parent_folder
            builder.template_remote_folder = template.outputs.remote_folder
            builder.template_metadata = orm.Dict(dict={
                "root_pk": self.state.get("root_pk"),
                "template_pk": template.pk,
                "template_uuid": template.uuid,
                "template_label": template.label,
            })
            builder.unfolding_parameters = orm.Dict(dict={
                "supercell_matrix": matrix.tolist(),
                "labels": labels,
                "path": coords,
                "npoints_per_segment": segment_counts,
                "kpoint_spacing": float(self.kpoint_spacing.value),
                "spinor": spin_settings["spinor"],
                "spin_channels": spin_settings["spin_channels"],
                "spin_mode": spin_settings["mode"],
            })
            if "settings" in template.inputs:
                builder.settings = template.inputs.settings
            if "parallelization" in template.inputs:
                builder.parallelization = template.inputs.parallelization
            builder.pw_metadata_options = orm.Dict(dict=self._resource_options(self.pw_resources))
            builder.banduppy_metadata_options = orm.Dict(dict=self._resource_options(self.banduppy_resources))
            node = submit(builder)
            self.folded_pk.value = node.pk
            html_status(self.status, "ok", f"Submitted QE BandUPpy unfolding WorkChain PK <code>{node.pk}</code>. Open the viewer with this PK after it finishes.")
            with self.details:
                print("Submitted WorkChain PK:", node.pk)
                print("WorkChain UUID:", node.uuid)
                print("Template PwCalculation PK:", template.pk)
                print("PW code:", builder.pw_code)
                print("BandUPpy code:", builder.banduppy_code)
                print("PW resources:", builder.pw_metadata_options.get_dict())
                print("BandUPpy resources:", builder.banduppy_metadata_options.get_dict())
                print("Spin mode:", spin_settings["description"])
                print("Path:", " - ".join(labels))
                print("Supercell matrix:")
                print(matrix)
        except Exception as exc:
            html_status(self.status, "err", f"<b>Submit failed:</b> {type(exc).__name__}: {exc}")
            with self.details:
                raise

    def refresh(self, _=None):
        pk = int(self.folded_pk.value or 0)
        if not pk:
            html_status(self.status, "info", "Enter a WorkChain PK or submit one.")
            return
        try:
            node = orm.load_node(pk)
            if getattr(node, "is_finished_ok", False):
                html_status(self.status, "ok", f"Unfolding WorkChain PK <code>{pk}</code> finished successfully. Plotting is available in the viewer.")
            elif getattr(node, "is_terminated", False):
                html_status(self.status, "err", f"WorkChain PK <code>{pk}</code> terminated unsuccessfully: {getattr(node, 'process_state', None)}")
            else:
                html_status(self.status, "warn", f"WorkChain PK <code>{pk}</code> is not finished yet: {getattr(node, 'process_state', None)}")
        except Exception as exc:
            html_status(self.status, "err", f"Cannot load PK <code>{pk}</code>: {type(exc).__name__}: {exc}")


class ViewerWidget(ipw.VBox):
    def __init__(self, pk=None, *, autoload=False):
        self.pk = ipw.IntText(value=int(pk or 0), description="Workflow PK", layout=ipw.Layout(width="280px"))
        self.load_button = ipw.Button(description="Load", icon="search", layout=ipw.Layout(width="100px"))
        self.plot_button = ipw.Button(description="Plot bands", button_style="success", icon="line-chart", disabled=True, layout=ipw.Layout(width="140px"))
        self.status = ipw.HTML()
        self.details = ipw.Output(layout=ipw.Layout(border="1px solid #ddd", padding="8px", max_height="260px", overflow="auto"))
        self.details_box = ipw.Accordion(children=[self.details], selected_index=None, layout=ipw.Layout(width="920px"))
        self.details_box.set_title(0, "Details")
        self.energy_min = ipw.FloatText(value=-6.0, description="E min", layout=ipw.Layout(width="150px"))
        self.energy_max = ipw.FloatText(value=6.0, description="E max", layout=ipw.Layout(width="150px"))
        self.threshold = ipw.FloatText(value=0.01, description="threshold", layout=ipw.Layout(width="180px"))
        self.fatfactor = ipw.IntText(value=20, description="fat", layout=ipw.Layout(width="130px"))
        self.spin_channel = ipw.Dropdown(options=[("default", "none")], value="none", description="spin", layout=ipw.Layout(width="190px"))
        self.retrieved = None
        self.node = None
        self.loaded = None
        self.load_button.on_click(self.load)
        self.plot_button.on_click(self.plot)
        super().__init__([
            ipw.HTML("<h3>View QE band unfolding</h3>"),
            ipw.HBox([self.pk, self.load_button, self.plot_button]),
            ipw.HBox([self.energy_min, self.energy_max, self.threshold, self.fatfactor, self.spin_channel]),
            self.status,
            self.details_box,
        ], layout=ipw.Layout(gap="8px"))
        if autoload and self.pk.value:
            self.load()

    def load(self, _=None):
        self.details.clear_output()
        self.plot_button.disabled = True
        self.loaded = None
        try:
            pk = int(self.pk.value)
            node = orm.load_node(pk)
            if not getattr(node, "is_finished_ok", False):
                raise RuntimeError(f"Node is not finished ok: {getattr(node, 'process_state', None)}")
            retrieved = self._get_banduppy_retrieved(node)
            with retrieved.open("unfolding_bands.npz", "rb") as handle:
                npz = np.load(handle)
                self.loaded = {key: npz[key] for key in npz.files}
            self.node = node
            self.retrieved = retrieved
            channels = _loaded_spin_channels(self.loaded)
            self.spin_channel.options = [("default", "none")] if channels == ["none"] else [(channel, channel) for channel in channels]
            self.spin_channel.value = self.spin_channel.options[0][1]
            self.spin_channel.disabled = len(channels) <= 1
            self.plot_button.disabled = False
            html_status(self.status, "ok", f"Loaded unfolding output from PK <code>{pk}</code>. Ready to plot.")
            with self.details:
                print("Node:", node.pk, node.__class__.__name__, process_label(node), getattr(node, "label", ""))
                print("Retrieved:", retrieved.pk)
                print("Files:", retrieved.base.repository.list_object_names())
                print("Available spin channels:", _loaded_spin_channels(self.loaded))
                print("Unfolded bandstructure shape:", self.loaded["unfolded_bandstructure"].shape)
                print("Supercell matrix:")
                print(self.loaded.get("supercell_matrix"))
        except Exception as exc:
            html_status(self.status, "err", f"<b>Load failed:</b> {type(exc).__name__}: {exc}")
            with self.details:
                print(f"Load failed: {type(exc).__name__}: {exc}")

    def _get_banduppy_retrieved(self, node):
        label = process_label(node)
        outputs = {triple.link_label: triple.node for triple in node.base.links.get_outgoing().all()}
        if label == "QeBanduppyUnfoldingWorkChain":
            if "banduppy_retrieved" not in outputs:
                raise RuntimeError("This WorkChain has no banduppy_retrieved output.")
            return outputs["banduppy_retrieved"]
        if label == "QeBanduppyCalculation":
            if "retrieved" not in outputs:
                raise RuntimeError("This BandUPpy calculation has no retrieved output.")
            return outputs["retrieved"]
        if label == "PwCalculation" and getattr(node, "label", "") == "BandUPpy folded-kpoints QE bands":
            raise RuntimeError(
                "PK points to the folded QE calculation. Open the completed "
                "QeBanduppyUnfoldingWorkChain PK instead."
            )
        raise RuntimeError(f"Unsupported node type for this viewer: {label or node.__class__.__name__}")

    def plot(self, _=None):
        try:
            import banduppy
            if self.loaded is None:
                self.load()
            data = self.loaded
            kline = _plot_kline(data, self.node)
            channel = self.spin_channel.value
            unfolded_raw = _unfolded_bandstructure_for_channel(data, channel)
            unfolded = _remap_unfolded_bandstructure_kline(unfolded_raw, kline)
            special_labels = json.loads(str(data.get("special_labels", "{}")))
            special_labels = _deduplicate_special_labels(
                {int(k): v for k, v in special_labels.items()}, kline
            )
            fermi = data.get("fermi_energy", np.nan)
            fermi = None if np.isnan(float(fermi)) else float(fermi)
            workdir = RESULTS_DIR / f"qe_unfolding_view_{self.node.pk}"
            workdir.mkdir(parents=True, exist_ok=True)
            log = io.StringIO()
            plot_unfold = banduppy.Plotting(save_figure_dir=str(workdir))
            fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=140)
            with contextlib.redirect_stdout(log):
                fig, ax, _ = plot_unfold.plot_ebs(
                    fig=fig,
                    ax=ax,
                    kpath_in_angs=kline,
                    unfolded_bandstructure=unfolded,
                    save_file_name=None,
                    CountFig=None,
                    Ef=fermi,
                    Emin=float(self.energy_min.value),
                    Emax=float(self.energy_max.value),
                    pad_energy_scale=0.5,
                    mode="fatband",
                    special_kpoints=special_labels,
                    plotSC=True,
                    marker="o",
                    fatfactor=int(self.fatfactor.value),
                    nE=100,
                    smear=0.2,
                    threshold_weight=float(self.threshold.value),
                    color="red",
                    color_map="viridis",
                    show_legend=True,
                    show_colorbar=False,
                    show_plot=False,
                )
            ax.set_ylabel(r"$E - E_F$ (eV)", fontsize=18)
            fig.tight_layout()
            outfile = workdir / "unfolded_bandstructure.png"
            fig.savefig(outfile, bbox_inches="tight", dpi=300)
            plt.show()
            html_status(self.status, "ok", f"Saved unfolded band plot: <code>{outfile}</code>")
            with self.details:
                captured = log.getvalue().strip()
                if captured:
                    print(captured)
                print("Plotted retrieved unfolding data from PK:", self.node.pk)
                print("Spin channel:", self.spin_channel.value)
                print("Saved plot:", outfile)
        except Exception as exc:
            html_status(self.status, "err", f"<b>Plot failed:</b> {type(exc).__name__}: {exc}")
            with self.details:
                raise


class SearchWidget(ipw.VBox):
    def __init__(self):
        self.refresh_button = ipw.Button(description="Search", icon="search", button_style="info", layout=ipw.Layout(width="130px"))
        self.results = ipw.HTML()
        self.refresh_button.on_click(self.search)
        super().__init__([
            ipw.HTML("<h3>Search QE post-processing calculations</h3>"),
            self.refresh_button,
            self.results,
        ], layout=ipw.Layout(gap="8px"))
        self.search()

    def search(self, _=None):
        from aiida.orm import QueryBuilder

        qb = QueryBuilder()
        qb.append(
            orm.WorkChainNode,
            filters={"attributes.process_label": "QeBanduppyUnfoldingWorkChain"},
            project=["id", "label", "attributes.process_state", "attributes.exit_status", "mtime"],
        )
        rows = sorted(qb.all(), key=lambda row: row[-1], reverse=True)[:30]
        if not rows:
            self.results.value = "<p>No QE BandUPpy unfolding workflows found yet.</p>"
            return

        body = []
        for pk, label, state, exit_status, mtime in rows:
            ready = _viewer_has_retrieved_unfolding(pk)
            finished_ok = str(state).lower().endswith("finished") and exit_status in (0, None)
            if ready:
                status = "ready"
            elif finished_ok:
                status = "missing retrieved data"
            else:
                status = "not finished ok"
            status_class = "qepp-ready" if ready else "qepp-warn"
            view_cell = (
                f'<a target="_blank" href="view_qe_unfolding.ipynb?pk={int(pk)}">View</a>'
                if ready
                else ""
            )
            body.append(
                "<tr>"
                f"<td>{int(pk)}</td>"
                f"<td>{escape(str(state or ''))}</td>"
                f"<td>{escape(str(exit_status if exit_status is not None else ''))}</td>"
                f"<td>{escape(str(mtime))}</td>"
                f"<td>{escape(str(label or 'BandUPpy unfolding workflow'))}</td>"
                f"<td class='{status_class}'>{status}</td>"
                f"<td>{view_cell}</td>"
                "</tr>"
            )
        self.results.value = (
            "<style>"
            ".qepp-search-table { border-collapse: collapse; min-width: 900px; max-width: 100%; font-size: 14px; }"
            ".qepp-search-table th, .qepp-search-table td { border: 1px solid #d8dee6; padding: 6px 8px; text-align: left; vertical-align: top; }"
            ".qepp-search-table th { background: #f3f6f8; font-weight: 650; }"
            ".qepp-search-table a { color: #0c6fb3; font-weight: 650; text-decoration: none; }"
            ".qepp-search-table a:hover { text-decoration: underline; }"
            ".qepp-ready { color: #1b5e20; }"
            ".qepp-warn { color: #8a5a00; }"
            "</style>"
            "<table class='qepp-search-table'>"
            "<thead><tr><th>Workflow PK</th><th>State</th><th>Exit</th><th>Modified</th><th>Label</th><th>Viewer state</th><th></th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody>"
            "</table>"
        )


def _loaded_spin_channels(data):
    if "spin_channels" in data:
        channels = [str(item) for item in np.asarray(data["spin_channels"]).tolist()]
        return channels or ["none"]
    channels = []
    for key in data:
        if key.startswith("unfolded_bandstructure_"):
            channels.append(key.removeprefix("unfolded_bandstructure_"))
    return sorted(channels) or ["none"]


def _unfolded_bandstructure_for_channel(data, channel):
    key = f"unfolded_bandstructure_{channel}"
    if channel != "none" and key in data:
        return data[key]
    if channel == "none" and "unfolded_bandstructure_none" in data:
        return data["unfolded_bandstructure_none"]
    return data["unfolded_bandstructure"]


def _plot_kline(data, node=None):
    if "primitive_kline" in data:
        return np.asarray(data["primitive_kline"], dtype=float)
    if "kpoints_pbz_full" in data:
        kpoints = np.asarray(data["kpoints_pbz_full"], dtype=float)[:, :3]
        if node is not None and "supercell_matrix" in data:
            try:
                structure = node.inputs.structure
                return _primitive_kline_from_cell(
                    structure.cell, np.asarray(data["supercell_matrix"], dtype=float), kpoints
                )
            except Exception:
                pass
        if len(kpoints) == 0:
            return np.array([], dtype=float)
        distances = np.linalg.norm(np.diff(kpoints, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(distances)))
    return np.asarray(data["kline"], dtype=float)


def _primitive_kline_from_cell(supercell_cell, supercell_matrix, fractional_kpoints):
    primitive_cell = np.linalg.solve(np.asarray(supercell_matrix, dtype=float), np.asarray(supercell_cell, dtype=float))
    reciprocal_cell = 2.0 * np.pi * np.linalg.inv(primitive_cell).T
    cartesian_kpoints = np.asarray(fractional_kpoints, dtype=float) @ reciprocal_cell
    if len(cartesian_kpoints) == 0:
        return np.array([], dtype=float)
    distances = np.linalg.norm(np.diff(cartesian_kpoints, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(distances)))


def _remap_unfolded_bandstructure_kline(unfolded_bandstructure, kline):
    remapped = np.asarray(unfolded_bandstructure, dtype=float).copy()
    if remapped.ndim != 2 or remapped.shape[1] < 4:
        return remapped
    indices = np.rint(remapped[:, 0]).astype(int)
    valid = (indices >= 0) & (indices < len(kline))
    remapped[valid, 1] = np.asarray(kline, dtype=float)[indices[valid]]
    return remapped


def _deduplicate_special_labels(special_labels, kline):
    result = {}
    seen = {}
    for index, label in sorted(special_labels.items()):
        if index < 0 or index >= len(kline):
            continue
        key = round(float(kline[index]), 10)
        previous = seen.get(key)
        if previous is None:
            seen[key] = index
            result[index] = label
        elif result[previous] != label:
            result[previous] = f"{result[previous]}/{label}"
    return result


def _viewer_has_retrieved_unfolding(pk):
    try:
        node = orm.load_node(int(pk))
        outputs = {triple.link_label: triple.node for triple in node.base.links.get_outgoing().all()}
        retrieved = outputs.get("banduppy_retrieved") or outputs.get("retrieved")
        if retrieved is None:
            return False
        return "unfolding_bands.npz" in retrieved.base.repository.list_object_names()
    except Exception:
        return False


def display_submission_widget():
    widget = SubmissionWidget()
    display(widget)
    return widget


def display_viewer_widget(pk=None, *, autoload=False):
    widget = ViewerWidget(pk=pk, autoload=autoload)
    display(widget)
    return widget


def display_search_widget():
    widget = SearchWidget()
    display(widget)
    return widget
