# AiiDAlab Empa QE post-processing

Lightweight AiiDAlab notebooks for Quantum ESPRESSO post-processing workflows developed at Empa.

Current prototype workflows:

- submit a BandUPpy-compatible folded-kpoints QE calculation for band unfolding;
- search completed post-processing calculations;
- visualize unfolded band structures from completed folded-kpoints QE calculations.

This is intentionally a notebook-style app, not yet a high-level `aiidalab-qe` plugin.

## Requirements

- Python 3.12 or newer and `aiida-core` 2.8 or newer;
- `aiida-nanotech-empa` with the `nanotech_empa.qe_banduppy` calculation and `nanotech_empa.qe.banduppy_unfolding` workflow entry points;
- configured Quantum ESPRESSO `pw.x` and BandUPpy AiiDA codes (the local development code is commonly labelled `banduppy-python@localhost`);
- a completed QE bands calculation to use as the supercell template.

The submission notebook launches the provenance-tracked unfolding workflow from `aiida-nanotech-empa`. The viewer loads its completed outputs and performs only plotting and lightweight result inspection.
