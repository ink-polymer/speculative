# r2 exact-source reproduction note

The final corrected H20 pilot manifest records this engine hash:

`1874a873ab3940ca4d1c58a3cc0f9fb222a3c8599fc5a0efeed9305e1011ed86`

The exact file is preserved as `r2_source/engine.py`.  It was reconstructed
byte-for-byte from the retained server preimage and the retained
`ddtree_fused_scan` integration patch; its SHA-256 matches the manifest.

The main-branch `src/gbv_experiments/engine.py` is a later integration
superset.  A direct diff shows that its additional changes only route
`DIFFUSION_SCAFFOLD_METHODS` through a lazy LM head.  The three r2 pilot
methods (`dflash`, `ddtree`, and `ddtree_fused_scan`) do not enter those
branches.  The exact snapshot is retained anyway so that source identity is
auditable rather than inferred.

The other four manifest sources still match the main tree byte-for-byte:

| Manifest key | Repository path | SHA-256 |
|---|---|---|
| `fused_verifier` | `src/gbv_experiments/fused_tree_sampling.py` | `ddec9c5a42f5fcc6f2254c8e5858f01c094c2f39090fdf6fe689dcc3e4d75ba8` |
| `official_verifier` | `src/gbv_experiments/sampling.py` | `43b246af3511f94a487fd462d8198ad574a71e8de407deb44f572ecbfe31a205` |
| `script` | `scripts/run_same_tree_official_pilot.py` | `3c7cbf0995664159a0bd4702bd10af02b445a98a9fb47031142620c8369caef3` |
| `config` | `configs/same_tree_block_qwen3_4b_pilot.json` | `9422b7114ddc70aa8d9065ef33132b1b90048eb26bbfdf07cfc6478478f311f6` |

Use `scripts/validate_same_tree_pilot.py` with `--engine-source` pointing at
the snapshot.  It checks all five source hashes and independently recomputes
the statistics from raw rows.
