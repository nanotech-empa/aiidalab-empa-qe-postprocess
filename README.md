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

## Plotting help

See the [unfolded-band plotting guide](docs/unfolded-band-plotting.md) for
path/state coverage, spectral styles, reference alignment and export. The viewer
also includes a collapsible **Plotting guide**.

## Manual GUI qualification

See [the bounded local guide](docs/testing/localhost.md), including its eight-atom
XYZ SCF-template route and explicit limits on conduction/duplicate coverage.

## Software attribution and scientific acknowledgements

Please credit [aiidalab-empa-qe-postprocess](https://github.com/nanotech-empa/aiidalab-empa-qe-postprocess) and its
[contributors](https://github.com/nanotech-empa/aiidalab-empa-qe-postprocess/graphs/contributors),
and identify the release or immutable commit used. Existing author,
contributor and funding acknowledgements remain authoritative.

See [CITATION.cff](CITATION.cff) for the software citation and identify the
exact release or commit used.

Preserve the copyright and permission notices in [LICENSE.txt](LICENSE.txt)
when copying or distributing copies or substantial portions of MIT-licensed
code. Separately licensed components retain their own notices. A name or
repository link does not replace the applicable licence notices.

For research using this app, acknowledge the [AiiDA team](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md#aiida)
and [AiiDAlab team](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md#aiidalab), and cite the three AiiDA papers
and both AiiDAlab platform papers listed in that guide.

Also credit aiida-nanotech-empa, aiida-quantumespresso, Quantum ESPRESSO and BandUPpy when used. These notebook workflows are distinct from the AiiDAlab QE app; using QE or its AiiDA plugin alone does not call for the QE-app paper.

The [shared credit guide](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md) keeps software attribution, licence
compliance and scientific citation requests separate. Scientific citation
requests are not additional MIT licence conditions.
