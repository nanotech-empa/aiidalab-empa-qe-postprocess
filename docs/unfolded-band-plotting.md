# Plotting unfolded bands

Open **View band unfolding**, enter a completed unfolding workflow PK, and click
**Load**. The **Plotting guide** beside the controls gives the same steps.

1. Start with **fat dots** and click **Plot bands**. The dots show calculated
   states and their unfolding weights without spectral smoothing. **threshold**
   hides weaker weights in fat dots and BandUPpy density; **fat** changes marker size.
2. Check what was calculated. The k-path and its sampling are chosen at submission.
   A handful of points on one segment is a short-path check, not a full
   high-symmetry dispersion. To inspect conduction bands, the QE calculation must
   include enough empty bands. Increasing **E max** cannot add missing states.
3. For presentation, compare **smoothed spectral density** and **BandUPpy density**
   with the dots. **smear** is the density styles' energy broadening in eV;
   **bins** controls their energy grid and **contrast** its intensity display
   (1 is linear). Sparse samples can produce broad horizontal-looking patches.
   Smoothing adds no calculated information.
4. To compare a primitive reference, select **reference primitive lines** and its
   **ref bands PK**. **ref shift** adds a rigid energy offset in eV to the reference
   curves only. This supports deliberate alignment, including comparisons of
   neutral and charged calculations. It does not change the unfolded eigenvalues,
   weights or stored calculation. Record the reference and the chosen shift when
   sharing the plot; matching Fermi zeros alone does not establish a common
   physical energy reference between different charged systems.
5. Click **Plot bands** again after changing controls. Download PNG or SVG from
   the resulting links. SVG preserves text/axes as vectors; spectral-density
   images are embedded at **export dpi**. Changing/loading a workflow clears the
   old plot and its live links. Existing exported files remain on disk.

A visually smooth plot is not a convergence test. Check the actual path,
computed energy range, numerical convergence and chosen energy alignment before
using it as a scientific band comparison.
