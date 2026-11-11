# AiiDAlab Empa QE post-processing

Lightweight AiiDAlab notebooks for Quantum ESPRESSO post-processing workflows developed at Empa.

Current prototype workflows:

- submit a BandUPpy-compatible folded-kpoints QE calculation for band unfolding;
- search completed post-processing calculations;
- visualize unfolded band structures from completed folded-kpoints QE calculations.

This is intentionally a notebook-style app, not yet a high-level `aiidalab-qe` plugin.

## Requirements

- Python 3.12 or newer, `aiida-core` >=2.8,<3 and AiiDAlab >=26.9.0;
- `aiida-nanotech-empa` with the `nanotech_empa.qe_banduppy` calculation and `nanotech_empa.qe.banduppy_unfolding` workflow entry points;
- configured Quantum ESPRESSO `pw.x` and BandUPpy AiiDA codes (the local development code is commonly labelled `banduppy-python@localhost`);
- a completed local QE bands calculation with its parent charge density, or a completed SCF template with `output_band` and its retained density/wavefunctions. An SCF-only template qualifies only the stored occupied/eigenvalue range, not missing conduction bands.

The submission notebook launches the provenance-tracked unfolding workflow from `aiida-nanotech-empa`. The viewer loads its completed outputs and performs only plotting and lightweight result inspection.

## Manual GUI qualification

See [the bounded local guide](docs/testing/localhost.md), including its eight-atom
XYZ SCF-template route and explicit limits on conduction/duplicate coverage.
