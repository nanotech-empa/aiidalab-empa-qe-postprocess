from __future__ import annotations

import ast
from html import escape
import json
import pickle
import re
from pathlib import Path

import aiidalab_widgets_base as awb
import aiidalab_widgets_empa as awe
import ipywidgets as ipw
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, PowerNorm
import numpy as np
from IPython.display import display
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import submit
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida_nanotech_empa.workflows.qe.banduppy import (
    _build_reference_primitive_structure,
    _reference_scf_kpoints,
)

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
        has_bands = outputs.get("output_band") is not None
        if has_bands:
            score = 2 + (1 if calculation == "bands" else 0)
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


def estimate_qe_band_window(calc, energy_min, energy_max, margin=2):
    outputs = outgoing_outputs(calc)
    bands_node = outputs.get("output_band")
    if bands_node is None:
        raise RuntimeError(
            f"QE PwCalculation PK {calc.pk} has no output_band. "
            "BandUPpy unfolding requires a finished QE bands calculation."
        )
    output_parameters = outputs.get("output_parameters")
    params = output_parameters.get_dict() if output_parameters is not None else {}
    fermi_energy = params.get("fermi_energy", 0.0)
    bands = np.asarray(bands_node.get_bands(), dtype=float)
    if bands.ndim < 1:
        raise RuntimeError(f"QE output_band for PK {calc.pk} contains no band data.")
    relative = bands - float(fermi_energy)
    axes = tuple(range(relative.ndim - 1))
    in_window = np.any(
        (relative >= float(energy_min)) & (relative <= float(energy_max)),
        axis=axes,
    )
    indices = np.where(in_window)[0]
    if len(indices) == 0:
        raise RuntimeError(
            "No QE bands fall inside the selected energy window. "
            "Widen the band energy range or increase the margin."
        )
    nbands = int(relative.shape[-1])
    ib_start = max(0, int(indices[0]) - int(margin))
    ib_end = min(nbands, int(indices[-1]) + int(margin) + 1)
    return {
        "ib_start": ib_start,
        "ib_end": ib_end,
        "first_band": ib_start + 1,
        "last_band": ib_end,
        "selected_bands": ib_end - ib_start,
        "total_bands": nbands,
        "fermi_energy": float(fermi_energy),
        "energy_min": float(energy_min),
        "energy_max": float(energy_max),
        "margin": int(margin),
    }


def format_band_window(window):
    if window is None:
        return "No band window selected."
    return (
        f"Bands {window['first_band']}-{window['last_band']} "
        f"({window['selected_bands']} of {window['total_bands']})\n"
        f"Energy window: {window['energy_min']:.3g} to {window['energy_max']:.3g} eV "
        f"relative to EF={window['fermi_energy']:.6g} eV, margin {window['margin']} bands"
    )


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
    if not candidates:
        raise RuntimeError("No finished QE PwCalculation with output_band found below this PK.")
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



def reference_preview_text(calc, matrix, tolerance):
    """Return a compact preview of the automatically reconstructed primitive reference."""
    structure = calc.inputs.structure
    reference = _build_reference_primitive_structure(
        structure, np.asarray(matrix, dtype=int), tolerance=float(tolerance)
    )
    parent_folder = calc.inputs.parent_folder if "parent_folder" in calc.inputs else None
    scf_kpoints = _reference_scf_kpoints(
        structure, reference, np.asarray(matrix, dtype=int), parent_folder
    )
    mesh, offset = scf_kpoints.get_kpoints_mesh()
    params = calc.inputs.parameters.get_dict()
    system = params.get("SYSTEM", {})
    original_charge = system.get("tot_charge", 0)
    original_nspin = system.get("nspin", 1)
    inv_reference_cell = np.linalg.inv(np.asarray(reference.cell, dtype=float))
    lines = [
        f"Primitive reference formula: {reference.get_formula()} ({len(reference.sites)} atoms)",
        f"PBC: {tuple(reference.pbc)}",
        f"Reference SCF mesh: {tuple(int(value) for value in mesh)}, offset {tuple(float(value) for value in offset)}",
        f"Reference charge/spin: neutral, nonmagnetic (template tot_charge={original_charge}, nspin={original_nspin})",
        "Primitive cell vectors (Ang):",
    ]
    lines.extend("  " + np.array2string(np.asarray(vector), precision=6, suppress_small=True) for vector in reference.cell)
    lines.append("Primitive atoms (fractional):")
    for site in reference.sites:
        frac = np.asarray(site.position, dtype=float) @ inv_reference_cell
        lines.append(
            f"  {site.kind_name:>4s} "
            + np.array2string(frac, precision=6, suppress_small=True)
        )
    try:
        description = json.loads(reference.description or "{}")
        clusters = description.get("clusters", [])
    except Exception:
        clusters = []
    if clusters:
        lines.append("Cluster counts used for ideal atoms:")
        for cluster in clusters:
            lines.append(f"  {cluster.get('kind')}: {cluster.get('counts')}")
    return "\n".join(lines)


class SubmissionWidget(ipw.VBox):
    def __init__(self):
        self.status = ipw.HTML()
        self.details = ipw.Output(layout=ipw.Layout(border="1px solid #ddd", padding="8px", max_height="260px", overflow="auto"))
        self.plot_output = ipw.Output(layout=ipw.Layout(width="100%"))
        self.details_box = ipw.Accordion(children=[self.details], selected_index=None, layout=ipw.Layout(width="920px"))
        self.details_box.set_title(0, "Details")
        self.pk = ipw.IntText(value=0, description="QE PK", layout=ipw.Layout(width="210px"))
        self.mode = ipw.ToggleButtons(options=[("S matrix", "matrix"), ("primitive lattice", "primitive")], value="matrix", description="Reference", layout=ipw.Layout(width="430px"))
        self.matrix = ipw.Text(value="[[2, 0, 0], [0, 2, 0], [0, 0, 2]]", description="S", layout=ipw.Layout(width="640px"))
        self.primitive = ipw.Textarea(value="", description="Primitive", layout=ipw.Layout(width="760px", height="70px"))
        self.run_reference = ipw.Checkbox(value=True, description="Run pristine primitive reference bands", indent=False)
        self.primitive_atom_tolerance = ipw.FloatText(value=0.08, description="atom tol", layout=ipw.Layout(width="180px"))
        self.reference_preview = ipw.Textarea(value="Load a QE calculation to preview the primitive reference cell.", description="preview", disabled=True, layout=ipw.Layout(width="900px", height="150px"))
        self.points_source = ipw.Dropdown(options=[("fcc preset", "fcc"), ("hexagonal 2D preset", "hex2d"), ("square 2D preset", "square2d"), ("rectangular 2D preset", "rect2d"), ("SeeK-path from selected structure", "seekpath"), ("custom", "custom")], value="fcc", description="Special k", layout=ipw.Layout(width="360px"))
        self.special_points = ipw.Textarea(value=format_points(SPECIAL_KPOINT_PRESETS["fcc"]), description="Available", layout=ipw.Layout(width="520px", height="150px"))
        self.path = ipw.Text(value="G X W K G L", description="Path", layout=ipw.Layout(width="640px"))
        self.kpoint_spacing = ipw.FloatText(value=0.1, description="spacing", layout=ipw.Layout(width="190px"))
        self.segment_counts = ipw.Textarea(value="Load a QE calculation to estimate points per segment.", description="segments", disabled=True, layout=ipw.Layout(width="760px", height="95px"))
        self.template = ipw.Dropdown(options=[], description="Template", layout=ipw.Layout(width="820px"))
        self.pw_code = _widgets_base_code_selector("QE pw code:", "quantumespresso.pw", preferred=("pw-7.4@localhost", "pw-7.4@daint.alps_lp83"))
        self.banduppy_code = _widgets_base_code_selector("BandUPpy code:", "nanotech_empa.qe_banduppy", preferred=("banduppy-python@localhost",))
        self.pw_resources = awe.ProcessResourcesWidget()
        self.reference_pw_resources = awe.ProcessResourcesWidget()
        self.banduppy_resources = awe.ProcessResourcesWidget()
        self.folded_diagonalization = ipw.Dropdown(
            options=[
                ("Davidson (robust)", "david"),
                ("inherit from template", "inherit"),
                ("ParO", "paro"),
                ("conjugate gradient", "cg"),
            ],
            value="david",
            description="QE diag",
            layout=ipw.Layout(width="260px"),
        )
        self.kpoint_batch_size = ipw.BoundedIntText(value=1, min=1, description="k/batch", layout=ipw.Layout(width="190px"))
        self.band_energy_min = ipw.FloatText(value=-6.0, description="band E min", layout=ipw.Layout(width="180px"))
        self.band_energy_max = ipw.FloatText(value=6.0, description="band E max", layout=ipw.Layout(width="180px"))
        self.band_margin = ipw.BoundedIntText(value=2, min=0, description="margin", layout=ipw.Layout(width="150px"))
        self.band_window = ipw.Textarea(value="Load a QE calculation to estimate the band window.", description="bands", disabled=True, layout=ipw.Layout(width="760px", height="58px"))
        self.pw_resources.walltime_seconds = 1800
        self.reference_pw_resources.walltime_seconds = 1800
        self.banduppy_resources.walltime_seconds = 1800
        self._set_resources_mpi(self.pw_resources, True)
        self._set_resources_mpi(self.reference_pw_resources, True)
        self._set_resources_mpi(self.banduppy_resources, False)
        self._set_resource_values(self.reference_pw_resources, nodes=1, tasks_per_node=1, threads_per_task=4)
        self._set_resource_values(self.banduppy_resources, nodes=1, tasks_per_node=1, threads_per_task=4)
        self.validate_button = ipw.Button(description="Load QE calculation", button_style="info", icon="search", layout=ipw.Layout(width="190px"))
        self.prepare_button = ipw.Button(description="Submit workflow", button_style="info", icon="cogs", layout=ipw.Layout(width="170px"))
        self.folded_pk = ipw.IntText(value=0, description="Workflow PK", layout=ipw.Layout(width="230px"))
        self.refresh_button = ipw.Button(description="Refresh", icon="refresh", layout=ipw.Layout(width="120px"))
        self.state = {}
        self.points_source.observe(self._set_points_from_source, names="value")
        for widget in (self.kpoint_spacing, self.path, self.special_points, self.matrix, self.primitive, self.mode):
            widget.observe(self._update_segment_counts, names="value")
        for widget in (self.matrix, self.primitive, self.mode, self.primitive_atom_tolerance):
            widget.observe(self._update_reference_preview, names="value")
        for widget in (self.band_energy_min, self.band_energy_max, self.band_margin):
            widget.observe(self._update_band_window, names="value")
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
            ipw.HTML("<b>Primitive reference QE resources</b>"),
            self.reference_pw_resources,
            self.banduppy_code,
            self.banduppy_resources,
            ipw.HBox([self.folded_diagonalization, self.kpoint_batch_size, ipw.HTML("<span style='line-height:32px;'>BandUPpy folded k-points loaded per batch</span>")]),
            ipw.HBox([self.band_energy_min, self.band_energy_max, self.band_margin, ipw.HTML("<span style='line-height:32px;'>Band window relative to EF</span>")]),
            self.band_window,
            ipw.HTML("<b>Reference cell</b>"),
            ipw.HBox([self.mode, self.matrix]),
            self.primitive,
            ipw.HBox([self.run_reference, self.primitive_atom_tolerance, ipw.HTML("<span style='line-height:32px;'>fractional tolerance for primitive atom clustering</span>")]),
            self.reference_preview,
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
    def _set_resource_values(resources, *, nodes=None, tasks_per_node=None, threads_per_task=None):
        for name, widget_name, value in (
            ("nodes", "nodes_widget", nodes),
            ("tasks_per_node", "tasks_per_node_widget", tasks_per_node),
            ("threads_per_task", "threads_per_task_widget", threads_per_task),
        ):
            if value is None:
                continue
            value = int(value)
            if hasattr(resources, name):
                try:
                    setattr(resources, name, value)
                except Exception:
                    pass
            widget = getattr(resources, widget_name, None)
            if widget is not None and hasattr(widget, "value"):
                try:
                    widget.value = value
                except Exception:
                    pass

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


    def _update_reference_preview(self, *_):
        template = self.state.get("template")
        if template is None:
            return
        try:
            structure = template.inputs.structure
            matrix, _, _ = self._current_reference_matrix(structure)
            self.reference_preview.value = reference_preview_text(
                template, matrix, self.primitive_atom_tolerance.value
            )
        except Exception as exc:
            self.reference_preview.value = f"Cannot preview primitive reference yet: {type(exc).__name__}: {exc}"

    def _update_band_window(self, *_):
        template = self.state.get("template")
        if template is None:
            return
        try:
            window = estimate_qe_band_window(
                template,
                self.band_energy_min.value,
                self.band_energy_max.value,
                self.band_margin.value,
            )
            self.state["band_window"] = window
            self.band_window.value = format_band_window(window)
        except Exception as exc:
            self.band_window.value = f"Cannot estimate band window yet: {type(exc).__name__}: {exc}"

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
            reference_preview = reference_preview_text(selected, matrix, self.primitive_atom_tolerance.value)
            self.reference_preview.value = reference_preview
            band_window = estimate_qe_band_window(
                selected,
                self.band_energy_min.value,
                self.band_energy_max.value,
                self.band_margin.value,
            )
            self.band_window.value = format_band_window(band_window)
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
                "reference_preview": reference_preview,
                "band_window": band_window,
            }
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
                print("Primitive reference preview:")
                print(reference_preview)
                print(f"Target k-point spacing: {float(self.kpoint_spacing.value):.4g} 1/Ang")
                print(f"BandUPpy k-point batch size: {int(self.kpoint_batch_size.value)}")
                print("Band window:", format_band_window(band_window))
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
            band_window = self.state.get("band_window")
            if band_window is None:
                raise RuntimeError("No QE band window is available; validate a finished QE bands calculation first.")

            builder = QeBanduppyUnfoldingWorkChain.get_builder()
            builder.pw_code = _load_code_from_widget(self.pw_code)
            builder.banduppy_code = _load_code_from_widget(self.banduppy_code)
            builder.run_reference_bands = orm.Bool(bool(self.run_reference.value))
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
            unfolding_parameters = {
                "supercell_matrix": matrix.tolist(),
                "labels": labels,
                "path": coords,
                "npoints_per_segment": segment_counts,
                "kpoint_spacing": float(self.kpoint_spacing.value),
                "kpoint_batch_size": max(1, int(self.kpoint_batch_size.value)),
                "folded_diagonalization": str(self.folded_diagonalization.value),
                "primitive_atom_tolerance": float(self.primitive_atom_tolerance.value),
                "spinor": spin_settings["spinor"],
                "spin_channels": spin_settings["spin_channels"],
                "spin_mode": spin_settings["mode"],
            }
            unfolding_parameters.update({
                "ib_start": int(band_window["ib_start"]),
                "ib_end": int(band_window["ib_end"]),
                "band_window_energy_min": float(band_window["energy_min"]),
                "band_window_energy_max": float(band_window["energy_max"]),
                "band_window_margin": int(band_window["margin"]),
            })
            builder.unfolding_parameters = orm.Dict(dict=unfolding_parameters)
            if "settings" in template.inputs:
                builder.settings = template.inputs.settings
            if "parallelization" in template.inputs:
                builder.parallelization = template.inputs.parallelization
            builder.pw_metadata_options = orm.Dict(dict=self._resource_options(self.pw_resources))
            builder.reference_pw_metadata_options = orm.Dict(dict=self._resource_options(self.reference_pw_resources))
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
                print("Folded QE resources:", builder.pw_metadata_options.get_dict())
                print("Primitive reference QE resources:", builder.reference_pw_metadata_options.get_dict())
                print("BandUPpy resources:", builder.banduppy_metadata_options.get_dict())
                print("Folded QE diagonalization:", self.folded_diagonalization.value)
                print("BandUPpy k-point batch size:", max(1, int(self.kpoint_batch_size.value)))
                print("Run primitive reference bands:", bool(self.run_reference.value))
                print("Primitive atom clustering tolerance:", float(self.primitive_atom_tolerance.value))
                print("Band window:", format_band_window(band_window))
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
        self.plot_output = ipw.Output(layout=ipw.Layout(width="100%"))
        self.details_box = ipw.Accordion(children=[self.details], selected_index=None, layout=ipw.Layout(width="920px"))
        self.details_box.set_title(0, "Details")
        self.energy_min = ipw.FloatText(value=-6.0, description="E min", layout=ipw.Layout(width="150px"))
        self.energy_max = ipw.FloatText(value=6.0, description="E max", layout=ipw.Layout(width="150px"))
        self.threshold = ipw.FloatText(value=0.01, description="threshold", layout=ipw.Layout(width="180px"))
        self.fatfactor = ipw.IntText(value=20, description="fat", layout=ipw.Layout(width="130px"))
        self.unfolding_plot_style = ipw.Dropdown(
            options=[("smoothed spectral density", "lines"), ("BandUPpy density", "density"), ("fat dots", "dots")],
            value="density",
            description="style",
            layout=ipw.Layout(width="230px"),
        )
        self.density_smear = ipw.FloatText(value=0.05, description="smear", layout=ipw.Layout(width="150px"))
        self.density_bins = ipw.IntText(value=240, description="bins", layout=ipw.Layout(width="140px"))
        self.density_contrast = ipw.FloatText(value=0.45, description="contrast", layout=ipw.Layout(width="160px"))
        self.spin_channel = ipw.Dropdown(options=[("default", "none")], value="none", description="spin", layout=ipw.Layout(width="190px"))
        self.qe_overlay = ipw.Dropdown(
            options=[
                ("supercell dots", "supercell"),
                ("reference primitive lines", "reference"),
                ("none", "none"),
            ],
            value="supercell",
            description="overlay",
            layout=ipw.Layout(width="280px"),
        )
        self.reference_bands_pk = ipw.IntText(value=0, description="ref bands PK", layout=ipw.Layout(width="250px"))
        self.reference_energy_shift = ipw.FloatText(value=0.0, description="ref shift", layout=ipw.Layout(width="160px"))
        self.reference_linewidth = ipw.FloatText(value=1.6, description="ref lw", layout=ipw.Layout(width="140px"))
        self.show_legend = ipw.Checkbox(value=True, description="legend", indent=False, layout=ipw.Layout(width="110px"))
        self.figure_width = ipw.FloatText(value=8.8, description="width [in]", layout=ipw.Layout(width="150px"))
        self.figure_height = ipw.FloatText(value=5.4, description="height [in]", layout=ipw.Layout(width="150px"))
        self.export_dpi = ipw.IntText(value=600, description="export dpi", layout=ipw.Layout(width="160px"))
        self.retrieved = None
        self.node = None
        self.loaded = None
        self.load_button.on_click(self.load)
        self.plot_button.on_click(self.plot)
        super().__init__([
            ipw.HTML("<h3>View QE band unfolding</h3>"),
            ipw.HBox([self.pk, self.load_button, self.plot_button]),
            ipw.HBox([self.energy_min, self.energy_max, self.threshold, self.fatfactor, self.spin_channel, self.unfolding_plot_style]),
            ipw.HBox([
                self.density_smear,
                self.density_bins,
                self.density_contrast,
                ipw.HTML("<span style='line-height:32px;'>spectral-density controls; contrast 1 = linear</span>"),
            ]),
            ipw.HBox([self.qe_overlay, self.reference_bands_pk, self.reference_energy_shift, ipw.HTML("<span style='line-height:32px;'>eV</span>"), self.reference_linewidth, self.show_legend]),
            ipw.HBox([
                self.figure_width,
                self.figure_height,
                self.export_dpi,
                ipw.HTML("<span style='line-height:32px;'>SVG keeps text and axes as vector; density maps are embedded at export dpi.</span>"),
            ]),
            self.status,
            self.details_box,
            self.plot_output,
        ], layout=ipw.Layout(gap="8px"))
        if autoload and self.pk.value:
            self.load()

    def load(self, _=None):
        self.details.clear_output()
        self.plot_output.clear_output(wait=True)
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
            if channels == ["none"]:
                self.spin_channel.options = [("default", "none")]
            else:
                options = [(channel, channel) for channel in channels]
                if len(channels) > 1:
                    options.append(("up + down", "both"))
                self.spin_channel.options = options
            self.spin_channel.value = self.spin_channel.options[0][1]
            self.spin_channel.disabled = len(channels) <= 1
            reference_bands = _workflow_reference_bands(node)
            if reference_bands is not None:
                self.reference_bands_pk.value = node.pk if process_label(node) == "QeBanduppyUnfoldingWorkChain" else reference_bands.pk
                self.qe_overlay.value = "reference"
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
        self.plot_output.clear_output(wait=True)
        try:
            if self.loaded is None:
                self.load()
            if self.loaded is None:
                raise RuntimeError("No unfolding data is loaded. Load a finished WorkChain PK first.")
            data = self.loaded
            kline = _plot_kline(data, self.node)
            selected_channels = _selected_spin_channels(data, self.spin_channel.value)
            special_labels = json.loads(str(data.get("special_labels", "{}")))
            special_labels = _deduplicate_special_labels(
                {int(k): v for k, v in special_labels.items()}, kline
            )
            fermi = data.get("fermi_energy", np.nan)
            fermi = None if np.isnan(float(fermi)) else float(fermi)
            workdir = RESULTS_DIR / f"qe_unfolding_view_{self.node.pk}"
            workdir.mkdir(parents=True, exist_ok=True)

            figure_width = max(2.0, float(self.figure_width.value))
            figure_height = max(2.0, float(self.figure_height.value))
            export_dpi = max(150, int(self.export_dpi.value))
            fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=140)
            overlay_text = "none"
            color_by_channel = {"none": "#d7191c", "up": "#d7191c", "dw": "#2c7bb6"}
            label_by_channel = {"none": "unfolded", "up": "unfolded up", "dw": "unfolded down"}
            density_cmap_by_channel = {"none": "Reds", "up": "Reds", "dw": "Blues"}
            plot_summaries = []
            if self.unfolding_plot_style.value == "density":
                for channel in selected_channels:
                    unfolded_raw = _unfolded_bandstructure_for_channel(data, channel)
                    unfolded = _remap_unfolded_bandstructure_kline(unfolded_raw, kline)
                    summary = _plot_unfolded_banduppy_density(
                        ax,
                        unfolded,
                        kline=kline,
                        fermi=fermi,
                        energy_min=float(self.energy_min.value),
                        energy_max=float(self.energy_max.value),
                        threshold=float(self.threshold.value),
                        smear=float(self.density_smear.value),
                        n_energy=max(20, int(self.density_bins.value)),
                        contrast=float(self.density_contrast.value),
                        cmap=density_cmap_by_channel.get(channel, "Reds"),
                        label=label_by_channel.get(channel, f"unfolded {channel}"),
                    )
                    plot_summaries.append(
                        f"{channel}: BandUPpy density map from {summary['points']} weighted points"
                    )

            if self.qe_overlay.value == "supercell":
                _plot_unfolded_supercell_dots(
                    ax,
                    data,
                    kline=kline,
                    fermi=fermi,
                    energy_min=float(self.energy_min.value),
                    energy_max=float(self.energy_max.value),
                    channels=selected_channels,
                )
                overlay_text = "BandUPpy supercell dots"
            elif self.qe_overlay.value == "reference":
                reference_info = _reference_bands_overlay(self.reference_bands_pk.value, self.node)
                reference_shift = float(self.reference_energy_shift.value)
                _plot_bandsdata_lines(
                    ax,
                    reference_info["bands"],
                    target_kline=kline,
                    target_special_labels=special_labels,
                    fermi=reference_info.get("fermi") if reference_info.get("fermi") is not None else fermi,
                    energy_min=float(self.energy_min.value),
                    energy_max=float(self.energy_max.value),
                    channels=selected_channels,
                    label="primitive reference",
                    energy_shift=reference_shift,
                    linewidth=max(0.1, float(self.reference_linewidth.value)),
                )
                overlay_text = f"primitive reference bands PK {reference_info['pk']} (own EF, shift {reference_shift:+.3f} eV)"
            if self.unfolding_plot_style.value != "density":
                for channel in selected_channels:
                    unfolded_raw = _unfolded_bandstructure_for_channel(data, channel)
                    unfolded = _remap_unfolded_bandstructure_kline(unfolded_raw, kline)
                    if self.unfolding_plot_style.value == "lines":
                        summary = _plot_unfolded_smoothed_density(
                            ax,
                            unfolded,
                            kline=kline,
                            special_labels=special_labels,
                            fermi=fermi,
                            energy_min=float(self.energy_min.value),
                            energy_max=float(self.energy_max.value),
                            smear=float(self.density_smear.value),
                            n_energy=max(20, int(self.density_bins.value)),
                            contrast=float(self.density_contrast.value),
                            cmap=density_cmap_by_channel.get(channel, "Reds"),
                            color=color_by_channel.get(channel, "#d7191c"),
                            label=label_by_channel.get(channel, f"unfolded {channel}"),
                        )
                        plot_summaries.append(
                            f"{channel}: smoothed spectral density from {summary['points']} weighted points "
                            f"across {summary['segments']} k-path segments"
                        )
                    else:
                        energies = unfolded[:, 2] - (0.0 if fermi is None else fermi)
                        weights = unfolded[:, 3]
                        mask = (
                            (energies >= float(self.energy_min.value))
                            & (energies <= float(self.energy_max.value))
                            & (weights >= float(self.threshold.value))
                        )
                        if not np.any(mask):
                            continue
                        sizes = 5.0 + 8.0 * float(self.fatfactor.value) * np.sqrt(np.clip(weights[mask], 0.0, None))
                        ax.scatter(
                            unfolded[mask, 1],
                            energies[mask],
                            s=sizes,
                            color=color_by_channel.get(channel, "#d7191c"),
                            alpha=0.72,
                            linewidths=0,
                            label=label_by_channel.get(channel, f"unfolded {channel}"),
                            zorder=3,
                        )
                        plot_summaries.append(f"{channel}: {int(np.sum(mask))} dots above threshold")

            _style_unfolding_axis(ax, kline, special_labels, float(self.energy_min.value), float(self.energy_max.value))
            handles, labels = ax.get_legend_handles_labels()
            if self.show_legend.value and handles:
                unique = dict(zip(labels, handles))
                ax.legend(
                    unique.values(),
                    unique.keys(),
                    loc="upper right",
                    fontsize=9,
                    framealpha=0.85,
                    handlelength=2.2,
                    borderpad=0.4,
                    labelspacing=0.35,
                )
            fig.tight_layout()
            outfile = workdir / "unfolded_bandstructure.png"
            svg_outfile = workdir / "unfolded_bandstructure.svg"
            with plt.rc_context(
                {
                    "svg.fonttype": "none",
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                    "savefig.facecolor": "white",
                }
            ):
                fig.savefig(outfile, bbox_inches="tight", dpi=300)
                fig.savefig(svg_outfile, bbox_inches="tight", dpi=export_dpi, format="svg")
            download_links = _viewer_download_links(
                [
                    ("Download PNG", outfile),
                    ("Download SVG", svg_outfile),
                ]
            )
            with self.plot_output:
                display(fig)
                display(ipw.HTML(download_links))
            plt.close(fig)
            html_status(
                self.status,
                "ok",
                "Saved unfolded band plot: "
                f"<code>{outfile}</code> and publication SVG: <code>{svg_outfile}</code>",
            )
            with self.details:
                print("Plotted retrieved unfolding data from PK:", self.node.pk)
                print("Spin channel:", self.spin_channel.value)
                print("Unfolding plot style:", self.unfolding_plot_style.value)
                print("Figure size [in]:", figure_width, figure_height)
                print("Export dpi:", export_dpi)
                print("Show legend:", bool(self.show_legend.value))
                print("Reference linewidth:", max(0.1, float(self.reference_linewidth.value)))
                for summary in plot_summaries:
                    print(summary)
                print("Overlay:", overlay_text)
                print("Saved plot:", outfile)
                print("Saved SVG:", svg_outfile)
        except Exception as exc:
            html_status(self.status, "err", f"<b>Plot failed:</b> {type(exc).__name__}: {exc}")
            with self.details:
                raise


def _viewer_download_links(items):
    links = []
    for label, path in items:
        try:
            relative_path = path.relative_to(APP_ROOT).as_posix()
            href = f"/files/apps/{APP_ROOT.name}/{relative_path}"
        except ValueError:
            href = path.as_posix()
        links.append(
            f'<a href="{escape(href)}" download>{escape(label)}</a>'
        )
    return (
        '<div style="margin:10px 0 4px 0; font-size:14px;">'
        + " &nbsp;|&nbsp; ".join(links)
        + "</div>"
    )


def _plot_unfolded_banduppy_density(
    ax,
    unfolded,
    *,
    kline,
    fermi,
    energy_min,
    energy_max,
    threshold,
    smear,
    n_energy,
    contrast,
    cmap,
    label,
):
    import contextlib
    import io

    import banduppy

    unfolded = np.asarray(unfolded, dtype=float)
    if unfolded.ndim != 2 or unfolded.shape[1] < 4 or len(unfolded) == 0:
        return {"points": 0}
    fermi = 0.0 if fermi is None else float(fermi)
    energies = unfolded[:, 2] - fermi
    weights = np.clip(unfolded[:, 3], 0.0, None)
    mask = (energies >= energy_min) & (energies <= energy_max) & (weights >= threshold)
    if not np.any(mask):
        return {"points": 0}

    before = len(ax.collections)
    plotter = banduppy.Plotting(save_figure_dir=str(RESULTS_DIR))
    kline = np.asarray(kline, dtype=float)
    unfolded_kpoints = np.column_stack((np.arange(len(kline)), kline))
    with contextlib.redirect_stdout(io.StringIO()):
        plotter.plot_ebs(
            unfolded_kpoints=unfolded_kpoints,
            unfolded_bandstructure=unfolded,
            fig=ax.figure,
            ax=ax,
            save_file_name=None,
            Ef=fermi,
            Emin=energy_min,
            Emax=energy_max,
            threshold_weight=threshold,
            mode="density",
            special_kpoints=None,
            plotSC=False,
            nE=max(20, int(n_energy)),
            smear=max(1.0e-6, float(smear)),
            color_map=cmap,
            show_legend=False,
            show_colorbar=False,
            show_plot=False,
            savefig=False,
        )
    for collection in ax.collections[before:]:
        density = np.asarray(collection.get_array(), dtype=float)
        finite_density = density[np.isfinite(density)]
        density_max = float(np.max(finite_density)) if finite_density.size else 0.0
        if density_max > 0.0:
            collection.set_norm(
                PowerNorm(
                    gamma=max(0.05, float(contrast)),
                    vmin=0.0,
                    vmax=density_max,
                )
            )
        collection.set_alpha(0.72)
        collection.set_rasterized(True)
        collection.set_zorder(0.2)
    ax.plot([], [], color=plt.get_cmap(cmap)(0.75), lw=4.0, alpha=0.72, label=label)
    return {"points": int(np.sum(mask))}


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


def _selected_spin_channels(data, value):
    channels = _loaded_spin_channels(data)
    if value == "both":
        return [channel for channel in channels if channel != "none"] or channels
    return [value]


def _outputs(node):
    return {triple.link_label: triple.node for triple in node.base.links.get_outgoing().all()}


def _input_dict(node):
    return {triple.link_label: triple.node for triple in node.base.links.get_incoming().all()}


def _pw_output_band(calc):
    if process_label(calc) != "PwCalculation":
        return None
    outputs = _outputs(calc)
    return outputs.get("output_band")


def _output_fermi(calc):
    params = _outputs(calc).get("output_parameters")
    if params is None:
        return None
    return params.get_dict().get("fermi_energy")


def _node_fermi(node):
    if process_label(node) == "PwCalculation":
        return _output_fermi(node)
    outputs = _outputs(node)
    params = outputs.get("output_parameters") or outputs.get("band_parameters")
    if params is not None:
        return params.get_dict().get("fermi_energy")
    for output in outputs.values():
        if output.__class__.__name__ == "Dict":
            value = output.get_dict().get("fermi_energy")
            if value is not None:
                return value
    return None


def _find_output_bands(node):
    if node.__class__.__name__ == "BandsData":
        return node
    outputs = _outputs(node)
    direct = outputs.get("output_band") or outputs.get("band_structure")
    if direct is not None and direct.__class__.__name__ == "BandsData":
        return direct
    for output in outputs.values():
        if output.__class__.__name__ == "BandsData":
            return output
    return None


def _workflow_reference_bands(node):
    workflow = node if process_label(node) == "QeBanduppyUnfoldingWorkChain" else None
    if workflow is None and process_label(node) == "QeBanduppyCalculation":
        folded_calc = _folded_qe_calc_from_viewer_node(node)
        workflow = _workflow_from_folded_qe_calc(folded_calc) if folded_calc is not None else None
    if workflow is None:
        return None
    outputs = _outputs(workflow)
    reference = outputs.get("reference_bands")
    return reference if reference is not None and reference.__class__.__name__ == "BandsData" else None


def _reference_bands_overlay(pk, viewer_node=None):
    pk = int(pk or 0)
    if pk <= 0 and viewer_node is not None:
        bands = _workflow_reference_bands(viewer_node)
        if bands is not None:
            creator = _creator(bands)
            fermi = _node_fermi(creator) if creator is not None else None
            return {"pk": bands.pk, "bands": bands, "fermi": fermi}
    if pk <= 0:
        raise RuntimeError("Enter a primitive reference BandsData/PwCalculation/WorkChain PK.")
    node = orm.load_node(pk)
    bands = _find_output_bands(node)
    if bands is None:
        raise RuntimeError(f"PK {pk} does not expose a BandsData output.")
    creator = _creator(bands)
    fermi = _node_fermi(node)
    if fermi is None and creator is not None:
        fermi = _node_fermi(creator)
    return {"pk": pk, "bands": bands, "fermi": fermi}


def _creator(node):
    incoming = node.base.links.get_incoming(link_type=LinkType.CREATE).all()
    return incoming[0].node if incoming else None


def _called_descendant_with_bands(node, *, folded=False):
    for child in descendants(node):
        if process_label(child) != "PwCalculation":
            continue
        if folded and getattr(child, "label", "") != "BandUPpy folded-kpoints QE bands":
            continue
        if _pw_output_band(child) is not None:
            return child
    return None


def _workflow_from_folded_qe_calc(calc):
    for triple in calc.base.links.get_incoming().all():
        if process_label(triple.node) == "QeBanduppyUnfoldingWorkChain":
            return triple.node
    return None


def _folded_qe_calc_from_viewer_node(node):
    if process_label(node) == "QeBanduppyUnfoldingWorkChain":
        return _called_descendant_with_bands(node, folded=True)
    if process_label(node) == "QeBanduppyCalculation":
        remote = _input_dict(node).get("folded_qe_remote_folder")
        if remote is not None:
            creator = _creator(remote)
            if _pw_output_band(creator) is not None:
                return creator
    return None


def _target_label_positions(special_labels, kline):
    positions = []
    for index, label in sorted(special_labels.items()):
        if index < 0 or index >= len(kline):
            continue
        position = float(kline[index])
        label = str(label).split("/")[0]
        if positions and abs(position - positions[-1][0]) < 1.0e-10:
            continue
        positions.append((position, canonical_label(label)))
    return positions


def _bandsdata_x_aligned(bands_node, target_kline, target_special_labels):
    bands = np.asarray(bands_node.get_bands(), dtype=float)
    nkpoints = int(bands.shape[-2])
    labels = [(int(index), canonical_label(label)) for index, label in getattr(bands_node, "labels", [])]
    target_positions = _target_label_positions(target_special_labels, target_kline)
    if len(labels) >= 2 and len(labels) == len(target_positions):
        label_names = [label for _, label in labels]
        target_names = [label for _, label in target_positions]
        if label_names == target_names:
            x = np.zeros(nkpoints, dtype=float)
            for (left_index, _), (right_index, _), (left_target, _), (right_target, _) in zip(
                labels[:-1], labels[1:], target_positions[:-1], target_positions[1:]
            ):
                if right_index <= left_index:
                    continue
                x[left_index:right_index + 1] = np.linspace(
                    left_target, right_target, right_index - left_index + 1
                )
            if labels[0][0] > 0:
                x[:labels[0][0]] = x[labels[0][0]]
            if labels[-1][0] < nkpoints - 1:
                x[labels[-1][0]:] = x[labels[-1][0]]
            return x
    if len(target_kline) == nkpoints:
        return np.asarray(target_kline, dtype=float)
    end = float(target_kline[-1]) if len(target_kline) else 1.0
    return np.linspace(0.0, end, nkpoints)


def _smooth_band_line(x, y, special_labels):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 4:
        return x, y
    boundaries = sorted({0, len(x) - 1, *[int(index) for index in special_labels if 0 <= int(index) < len(x)]})
    smooth_x = []
    smooth_y = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left:
            continue
        segment_x = x[left : right + 1]
        segment_y = y[left : right + 1]
        keep = np.concatenate(([True], np.diff(segment_x) > 1.0e-12))
        segment_x = segment_x[keep]
        segment_y = segment_y[keep]
        if len(segment_x) < 3 or segment_x[-1] <= segment_x[0]:
            interp_x = segment_x
            interp_y = segment_y
        else:
            interp_x = np.linspace(segment_x[0], segment_x[-1], max(len(segment_x), 12 * (len(segment_x) - 1) + 1))
            try:
                from scipy.interpolate import PchipInterpolator

                interp_y = PchipInterpolator(segment_x, segment_y)(interp_x)
            except Exception:
                interp_y = np.interp(interp_x, segment_x, segment_y)
        if smooth_x and len(interp_x) and abs(float(interp_x[0]) - float(smooth_x[-1])) < 1.0e-12:
            interp_x = interp_x[1:]
            interp_y = interp_y[1:]
        smooth_x.extend(interp_x.tolist())
        smooth_y.extend(interp_y.tolist())
    if not smooth_x:
        return x, y
    return np.asarray(smooth_x), np.asarray(smooth_y)


def _plot_bandsdata_lines(
    ax,
    bands_node,
    *,
    target_kline,
    target_special_labels,
    fermi,
    energy_min,
    energy_max,
    channels,
    label="QE bands",
    energy_shift=0.0,
    linewidth=0.95,
):
    bands = np.asarray(bands_node.get_bands(), dtype=float)
    if bands.ndim == 2:
        bands = bands[np.newaxis, :, :]
    x = _bandsdata_x_aligned(bands_node, target_kline, target_special_labels)
    fermi = 0.0 if fermi is None else float(fermi)
    channel_to_spin = {"none": [0], "up": [0], "dw": [1]}
    if "both" in channels:
        spin_indices = list(range(bands.shape[0]))
    else:
        spin_indices = []
        for channel in channels:
            spin_indices.extend(channel_to_spin.get(channel, [0]))
        spin_indices = sorted({index for index in spin_indices if index < bands.shape[0]}) or [0]
    line_colors = ["#7b5bb7", "#4f74c8"]
    for spin_index in spin_indices:
        rel = bands[spin_index] - fermi + float(energy_shift)
        color = line_colors[spin_index % len(line_colors)]
        first = True
        for band_index in range(rel.shape[1]):
            energies = rel[:, band_index]
            if np.nanmax(energies) < energy_min or np.nanmin(energies) > energy_max:
                continue
            plot_x, plot_y = _smooth_band_line(x, energies, target_special_labels)
            ax.plot(
                plot_x,
                plot_y,
                color=color,
                lw=max(0.1, float(linewidth)),
                alpha=0.58,
                antialiased=True,
                zorder=2,
                label=label if first else None,
            )
            first = False



def _plot_unfolded_smoothed_density(
    ax,
    unfolded,
    *,
    kline,
    special_labels,
    fermi,
    energy_min,
    energy_max,
    smear,
    n_energy,
    contrast,
    cmap,
    color,
    label,
):
    unfolded = np.asarray(unfolded, dtype=float)
    if unfolded.ndim != 2 or unfolded.shape[1] < 4 or len(unfolded) == 0:
        return {"points": 0, "segments": 0}
    kline = np.asarray(kline, dtype=float)
    if len(kline) < 2:
        return {"points": 0, "segments": 0}

    fermi = 0.0 if fermi is None else float(fermi)
    k_indices = np.rint(unfolded[:, 0]).astype(int)
    energies = unfolded[:, 2] - fermi
    weights = np.clip(unfolded[:, 3], 0.0, None)
    smear = max(1.0e-6, float(smear))
    energy_grid = np.linspace(float(energy_min), float(energy_max), max(20, int(n_energy)))
    finite = np.isfinite(energies) & np.isfinite(weights)
    mask = (
        finite
        & (k_indices >= 0)
        & (k_indices < len(kline))
        & (weights > 0.0)
        & (energies >= float(energy_min) - 5.0 * smear)
        & (energies <= float(energy_max) + 5.0 * smear)
    )
    if not np.any(mask):
        return {"points": 0, "segments": 0}

    density = np.zeros((len(kline), len(energy_grid)), dtype=float)
    for index, energy, weight in zip(k_indices[mask], energies[mask], weights[mask]):
        density[index] += weight * np.exp(
            -0.5 * ((energy_grid - energy) / smear) ** 2
        )

    # Smooth the intensity field independently inside each path segment.  This
    # avoids inventing a band-to-band correspondence and prevents interpolation
    # across duplicated high-symmetry endpoints.
    boundaries = {
        0,
        len(kline) - 1,
        *[
            int(index)
            for index in special_labels
            if 0 <= int(index) < len(kline)
        ],
    }
    for index in np.where(np.diff(kline) <= 1.0e-12)[0]:
        boundaries.update((int(index), int(index + 1)))
    boundaries = sorted(boundaries)

    from scipy.interpolate import PchipInterpolator

    smooth_segments = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left or kline[right] <= kline[left]:
            continue
        source_x = kline[left : right + 1]
        source_density = density[left : right + 1]
        unique_x, inverse = np.unique(source_x, return_inverse=True)
        if len(unique_x) < 2:
            continue
        unique_density = np.zeros((len(unique_x), len(energy_grid)), dtype=float)
        counts = np.zeros(len(unique_x), dtype=float)
        np.add.at(unique_density, inverse, source_density)
        np.add.at(counts, inverse, 1.0)
        unique_density /= counts[:, np.newaxis]

        dense_count = max(len(unique_x), 12 * (len(unique_x) - 1) + 1)
        dense_x = np.linspace(unique_x[0], unique_x[-1], dense_count)
        if len(unique_x) >= 3:
            smooth_density = PchipInterpolator(
                unique_x, unique_density, axis=0
            )(dense_x)
        else:
            fraction = (
                (dense_x - unique_x[0]) / (unique_x[-1] - unique_x[0])
            )[:, np.newaxis]
            smooth_density = (
                (1.0 - fraction) * unique_density[0]
                + fraction * unique_density[-1]
            )
        smooth_segments.append((dense_x, np.clip(smooth_density, 0.0, None)))

    if not smooth_segments:
        return {"points": int(np.sum(mask)), "segments": 0}

    density_max = max(float(np.max(values)) for _, values in smooth_segments)
    if density_max <= 0.0:
        return {"points": int(np.sum(mask)), "segments": 0}

    base_cmap = plt.get_cmap(cmap)
    colors = base_cmap(np.linspace(0.25, 1.0, 256))
    colors[:, 3] = np.linspace(0.0, 0.92, len(colors))
    transparent_cmap = ListedColormap(colors)
    background_cutoff = 1.0e-3
    gamma = max(0.05, float(contrast))
    for dense_x, smooth_density in smooth_segments:
        relative_density = smooth_density / density_max
        visible_density = np.clip(
            (relative_density - background_cutoff) / (1.0 - background_cutoff),
            0.0,
            1.0,
        ) ** gamma
        collection = ax.pcolormesh(
            dense_x,
            energy_grid,
            visible_density.T,
            cmap=transparent_cmap,
            vmin=0.0,
            vmax=1.0,
            shading="gouraud",
            rasterized=True,
            zorder=0.2,
        )
        collection.set_edgecolor("none")

    ax.plot([], [], color=color, lw=4.0, alpha=0.78, label=label)
    return {"points": int(np.sum(mask)), "segments": len(smooth_segments)}


def _plot_unfolded_supercell_dots(ax, data, *, kline, fermi, energy_min, energy_max, channels):
    fermi = 0.0 if fermi is None else float(fermi)
    label_used = False
    for channel in channels:
        unfolded = _remap_unfolded_bandstructure_kline(
            _unfolded_bandstructure_for_channel(data, channel), kline
        )
        energies = unfolded[:, 2] - fermi
        mask = (energies >= energy_min) & (energies <= energy_max)
        if not np.any(mask):
            continue
        ax.scatter(
            unfolded[mask, 1],
            energies[mask],
            s=13,
            color="0.45",
            alpha=0.42,
            linewidths=0,
            label=None if label_used else "supercell",
            zorder=1,
        )
        label_used = True


def _style_unfolding_axis(ax, kline, special_labels, energy_min, energy_max):
    ax.axhline(0.0, color="black", lw=0.9, ls="--", alpha=0.8)
    ticks = []
    ticklabels = []
    for index, label in sorted(special_labels.items()):
        if index < 0 or index >= len(kline):
            continue
        position = float(kline[index])
        if ticks and abs(position - ticks[-1]) < 1.0e-10:
            ticklabels[-1] = f"{ticklabels[-1]}/{label}"
            continue
        ticks.append(position)
        ticklabels.append(str(label))
        ax.axvline(position, color="black", lw=0.9, alpha=0.65)
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
    if len(kline):
        ax.set_xlim(float(np.nanmin(kline)), float(np.nanmax(kline)))
    ax.set_ylim(energy_min, energy_max)
    ax.set_xlabel("k-points", fontsize=11)
    ax.set_ylabel(r"$E - E_F$ (eV)", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)


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
