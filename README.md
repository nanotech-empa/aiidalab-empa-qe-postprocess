# AiiDAlab Empa QE post-processing

Lightweight AiiDAlab notebooks for Quantum ESPRESSO post-processing workflows developed at Empa.

Current prototype workflows:

- submit a BandUPpy-compatible folded-kpoints QE calculation for band unfolding;
- search completed post-processing calculations;
- visualize unfolded band structures from completed folded-kpoints QE calculations.

This is intentionally a notebook-style app, not yet a high-level `aiidalab-qe` plugin.

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
