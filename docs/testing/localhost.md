# Empa QE post-processing — local manual BandUPpy qualification

This is the separate `aiidalab-empa-qe-postprocess` notebook app, not Surfaces'
CP2K unfolding and not a property panel of the main QE app. Use current canonical
main with the merged backend/shared-component pins. No case starts before the
unified readiness report, actual installation preflight PASS and user start.
The operator submits only in the real browser; maintainers audit read-only.
XPS, remote compute and spinor/magnetic/defective unfolding are excluded here.

Record exact app/backend/main-QE commits, installed distributions/import origins,
Code/Computer UUIDs, executable/pseudo hashes and source/preflight fingerprints.
Changing relevant source invalidates affected qualification. Preserve old reports
and exports; old campaign nodes deleted by authorized cleanup are historical
evidence, not fixtures that can still be reopened. No previous BandUPpy science
PASS is implied by these prepared instructions.

## Model, route and budgets

Use [pristine 2×2 hBN](fixtures/hbn-2x2-xyz.extxyz), 8 atoms (4B4N), neutral,
nonmagnetic PBE, a=b=5 Å, gamma120°, c=15 Å. Primitive a=2.5 Å, two-atom basis;
S=diag(2,2,1). This is deliberately a **3D-periodic XYZ repeated-slab model**
with vacuum, matching the campaign's CP2K slab model. A hexagonal 2D *path preset*
does not change QE electrostatics. Do not silently use 2D Coulomb isolation,
change PBC, introduce distinct atom kinds, distort the cell or reduce to 2 atoms
merely to avoid SeeK-path. A deliberate different model needs its own recorded
scientific choice and corresponding CP2K comparison, not an unnoticed substitute.

The main QE Bands workflow can reduce pristine XYZ supercells. The first small
route therefore uses its **SCF-only** GUI workflow: Structure as is, no optional
properties. The app includes the non-optimizing relax/SCF stage in that case.
Its source structure remains eight atoms. This companion app accepts completed
SCFs with output_band and uses their own retained density as the charge source.
Existing bands templates retain their actual input parent. Other parentless
calculation types are explicitly rejected. SCF-only templates contain the stored
occupied/eigenvalue range; **conduction bands and a band gap are not qualified**
without a separately supported GUI template containing those states. Do not
invent an editable 20-band control absent from the main QE form.

Keep one native job at a time. QE panels: one node, **2 MPI tasks/node × 1
thread/task = 2 cores**, walltime600s. BandUPpy: one process/one thread, **k/batch
=1**, walltime600s. Set all three resource panels explicitly; initial constructor
values are not an agreed budget. No memory limit field is assumed. Observe RAM,
free disk and job-tree work size every 30 s; initial allowance 4 GiB new work and
250 MiB retrieved across the case. No extra BLAS gate is imposed. Use the agreed
exact-root stop route if a serious fault or stage ceiling is reached. No restart,
retry or larger machine is automatic. Tiny QE cost is unmeasured; the proposed
cap is ten minutes per stage. The route has at most **five serial native jobs**:
one template SCF, reference SCF, reference bands, folded QE bands, BandUPpy.

PASS/FAIL/BLOCKED criteria apply separately to GUI integration and physical
validation. Native Finished0 or a nonempty NPZ is necessary, not scientific proof.
DOC01 below applies to all cases.

## B00 — Companion entry and local prerequisites (no jobs)

1. Open the maintained launcher. Require **QE post-processing**, **Submit band
   unfolding** and **Search post-processing**. The canonical entry is
   `submit_qe_unfolding.ipynb`; a guessed Voila route is not a prerequisite.
2. Inspect **QE PK**, **Load QE calculation**, **Template**, **QE pw code:**,
   **BandUPpy code:**, three resource panels, **Reference**, **Special k**,
   **Path**, **Submit workflow**, **Workflow PK**, **Refresh** and **Details**.
3. Require qualified local pw.x and BandUPpy codes. Record versions/plugins,
   computers and actual executable/wrapper. Registration is not executable
   installation or native API qualification. Missing code/app is BLOCKED.
4. Opening, selecting, loading or plotting must submit nothing. Record layout,
   scrolling and help findings without redesigning the established interface.
5. Scroll the complete submission form from the template/code resources to the
   path and Details, then back. Forms, diagnostic output and result plots must
   expand in the normal page flow, using the page scrollbar. Input text editors
   retain their own scrolling. Confirm resource selections survive page scrolling.

## B01 — Eight-atom template through the main QE GUI

Needs B00, admitted resources and the user's test start. One SCF job maximum.

1. In the main QE GUI upload the fixture through its real structure selector.
   Confirm 4B4N/eight atoms, cell lengths 5,5,15 Å, gamma120° and XYZ PBC.
2. Choose **Structure as is**. Deselect optional property calculations, including
   Bands/PDOS/phonons. Source inspection shows this executes the non-optimizing
   SCF stage; require the actual preview/builder to confirm it before submission.
   **Basic settings:** Insulator, Spin Off, Spin-orbit Off, Balanced PBE protocol.
   Keep scientifically supported pseudo cutoffs, no artificial cutoff reduction.
3. In Advanced convergence inspect **K-points distance** and the resulting grid.
   A candidate distance 2 Å⁻¹ should yield 1×1×1 for this cell; require the actual
   Γ-only mesh/offset preview instead of inferring it. Retain workfiles: keep
   **Delete the work directory after the calculation** OFF. Record selected B/N
   pseudos, valences and cutoffs. Ordinary B(3)/N(5) valence gives 32 electrons and 16
   occupied nonmagnetic bands; verify the actual pseudo metadata/count.
4. Select localhost pw.x, 1 node / 2 MPI / 1 thread / 600 s, recheck all controls and submit
   once. Record parent/native UUIDs. Require native calculation=scf, unchanged
   eight-atom structure/cell, convergence and output_band. No ionic relaxation,
   SeeK-path primitive substitution or unexpected property job may occur.
5. Independently verify actual density/wavefunction files in the local remote
   folder, input recipe, eigenvalues/occupations and output_band identity.
   An imported RemoteData without files is insufficient. No SCF rerun/terminal
   fixture is allowed to conceal a missing GUI route. If controls or preview do
   not establish this route, mark BLOCKED and retain the draft without submission.
6. In the companion app enter that parent/native **QE PK**, click **Load QE
   calculation**, and select the intended **Template** if multiple candidates
   exist. Require its exact native UUID/atom count; SCF-template support must
   carry this SCF's converged charge folder to folded bands.

## B02 — Mapping, current form and invalid drafts (no jobs)

1. Choose **Reference → S matrix**, `[[2,0,0],[0,2,0],[0,0,1]]`, **Special k →
   hexagonal 2D preset**, **Path = G M**, spacing 0.4 Å⁻¹. Load/validate and inspect
   segment counts; require at most six primitive path points. Reconstruct the
   row-vector primitive/supercell cells and reciprocal distances explicitly.
2. Keep **Run pristine primitive reference bands** ON. Require the preview's
   two-atom BN primitive geometry, XYZ model and neutral nonmagnetic recipe.
   Atom tol 0.08 is a fractional clustering tolerance, not Å or an error bound.
3. Select Davidson (robust), k/batch 1. Initially choose band E min/max −20,+10 eV
   relative to the displayed EF, margin 2; inspect the inferred *stored* indices.
   Narrow only after identifying the occupied states. An energy window neither
   creates missing bands nor reduces inherited nbnd/cutoffs. Record whether EF
   is an actual reported value or a fallback; this matters for energy alignment.
4. Set all three budgets explicitly. Change path/spacing to another valid draft,
   reload, check current counts, then restore. Try a singular S matrix, undefined
   path, nonpositive spacing and wrong source only as validation controls.
   Require clear rejection and no stale accepted state. Current submit rereads
   the visible form; unit tests alone do not supply this GUI PASS. Restore and
   validate the valid final form before any scientific submission.
   After loading a valid draft, change spacing to zero and test **Submit workflow**:
   require the finite-positive-spacing diagnostic and no new workflow, then
   restore 0.4 and load again. Invalid spacing must never become a tiny default
   spacing or an enormous point count.
5. The equivalent explicit primitive-lattice representation may be compared as
   a draft-only mapping check. Restore the chosen representation afterward.

## B03 — One serial workflow and scientific unfolding

Needs B02. Four native stages, strictly serial. Press **Submit workflow** once;
record Workflow PK/UUID, source UUID and every child UUID from Details/provenance.
Use **Refresh** on this exact workflow; no second submission to monitor it.

1. Verify reference SCF → reference bands → folded QE bands → BandUPpy order,
   convergence, actual code/computer/2 MPI / 1 thread layouts and 600 s limits. Inspect
   inherited parallelization options so pools do not request additional ranks.
2. Compare stored folded_kpoints, mapping_arrays/mapping_data, primitive structure
   and reference sampling with the form. Check cell/reciprocal conventions,
   state count, spin/band indices and retained native data completeness.
3. Audit finite unfolded energies/nonnegative weights (proposed tolerance 1e-6).
   For pristine hBN compare common occupied bands at identical primitive q to
   the reference: propose 0.02 eV after a *justified* common energy zero. Freeze
   criteria first. Treat crossings/degeneracies as subspaces; do not force
   individual band labels. Insulator EF conventions may differ: no fitted
   unexplained shift simply to make the overlay agree.
4. Sum weight only over a complete primitive-q representative set for the same
   supercell state if the stored mapping actually contains it. A short G–M path
   is not generally complete. Never renormalize partial weights to manufacture
   a unit sum. Report an unavailable sum-rule check as unvalidated.
5. Split functional execution/visualization PASS from scientific mapping/weight/
   reference agreement. Missing conduction states leave those branches explicitly
   unvalidated. A CP2K unfolding result cannot substitute for this QE evidence.

If the three QE stages succeeded but the unfolding reader failed, preserve the
failed workflow and all successful remote folders. After a reviewed reader-only
correction and source/runtime preflight, the existing GUI recovery route can use
the completed folded `PwCalculation` as the template with **Run pristine primitive
reference bands** OFF. It submits folded bands followed by BandUPpy, with no SCF
or repeated primitive reference. Keep the same S/path/band selection and explicit
resources. Record this as a new recovery workflow; do not relabel the failed run.
In the viewer, select **reference primitive lines** and the earlier primitive
reference BandsData in **ref bands PK**. Audit that it is the same Hamiltonian,
cell/path/spin recipe and common energy zero. This route needs its own actual
GUI/native evidence; a reader unit test does not qualify the recovered spectrum.

## B04 — Search, viewer, export, stale state and reopening

No native jobs. Search post-processing → Search → View this workflow, or use
`view_qe_unfolding.ipynb?pk=<recorded-PK>`. Search lists the newest 30; save the
actual durable UUID/direct reopening route, not an invented UUID filter.

1. Use Workflow PK → Load → Plot bands. Compare fat dots, BandUPpy density and
   smoothed density. Change E range, threshold, fat/style, broadening and spin
   controls, then restore. Density broadening is display, not lifetime data.
2. Overlay reference primitive lines from this run's own reference. Start ref
   shift 0 and audit each EF convention as in B03. No extra calculation occurs.
   Stored BandUPpy energies are already E−EF; require all three plot styles and
   supercell dots to use that zero exactly once. Change **ref shift** positively
   and negatively, check that only the primitive reference moves by that amount,
   then restore 0. This manual offset remains available for charged/neutral
   comparisons; it must not compensate for an automatic duplicate EF subtraction.
3. Download actual PNG and SVG. Check editable SVG text/axes, raster density at
   chosen export dpi, labels, dimensions and data identity against raw arrays.
   Preserve raw NPZ independently; do not claim a nonexistent raw-data GUI export.
4. Load another authorized result or invalid PK; require stale arrays/plot/export
   feedback to clear or be explicitly stale. Never delete original files to
   manufacture failures. Close/reopen a fresh page through Search and require
   identical source UUIDs/arrays/plots with no job.
5. Duplicate protection is separate: current direct submit has no established
   durable lock. Do not double-submit real jobs to prove that. Record this limit;
   a future safe guard/retest must independently establish exactly one parent.

## DOC01, comparison and tab lifecycle

Review help for the SCF template/charge source, retained files, XYZ slab model,
S convention, primitive clustering, reciprocal units, band/EF range, spin/reference
choices, resources and spectral weights. Identify actual maintained help used by
AiiDAlab/in-app assistants; prepared docs alone do not prove chatbot retrieval.
Preserve BandUPpy, QE and AiiDA citations. See [BandUPpy](https://github.com/band-unfolding/banduppy).

CP2K 6×6 Γ-only folds 36 q points; it cannot supply an arbitrary continuous QE G–M
line. Compare only matching Γ/M/etc coordinates after each track has independent
validation; methods, bases/pseudos and energy zeros differ. No unagreed absolute
cross-code eigenvalue tolerance is implied. CP2K's 8 MPI case does not change QE's
selected2 MPI / 1 thread budget.

After each completed manual GUI case, record the verdict and reopen route, then
close its Chrome AiiDAlab tabs. Keep active/unfinished/pending-user-review and
unrelated tabs. Closing tabs must not cancel jobs or shared services. Preserve
all failed/partial evidence and mark unsupported/unconfigured branches honestly.
