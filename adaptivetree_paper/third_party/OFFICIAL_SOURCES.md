# Packaged reference source

This AdaptiveTree-only package includes `ddtree_official/ddtree.py` as a
reference for the DDTree builder equivalence tests. The tests extract the pure
builder function; the complete upstream benchmark is not shipped or run.

The workspace records the upstream source as
https://github.com/liranringel/ddtree at commit
`c96427a185677bf4133ed865dd1626a5041aef9b`.
The actual copied file is identified by its SHA-256 in
`../SOURCE_SHA256.json`; its existing MIT license is retained at
`ddtree_official/LICENSE`.

The runtime uses the repository's controlled DFlash/target adapters and downloads
the frozen original model revisions specified in `configs/paper_t0_model.json`.
Model weights and benchmark datasets are not included. This package contains no
GBV experiment suite and no new formal performance results.
