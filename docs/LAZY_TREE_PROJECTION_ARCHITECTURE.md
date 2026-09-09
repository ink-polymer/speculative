# Lazy-projection tree block decoding

## Claim boundary

This architecture has a strict expected-work advantage over the repository's
DDTree baseline for the vocabulary projection and posterior-sampling portion
of a decode round.  It does not claim that lower operation count guarantees a
wall-clock win on every GPU.  The Target transformer, tree construction, mask,
and cache compaction remain common costs and must be measured end to end.

## Architecture

Let a verified probability tree contain `N = B + 1` nodes, partitioned into
`I` internal nodes and `F` leaves, so `N = I + F`.

The DDTree baseline applies the Target LM head to all `N` hidden-state rows,
constructs all `N` vocabulary distributions, samples all rows in one batch,
and discards draws that are not on the realized ancestral path.

`ddtree_lazy_projection` instead:

1. runs the identical Target transformer and produces the same `N` hidden rows;
2. applies the LM head to all `I` internal rows as one batch;
3. samples those internal rows as one batch and follows the realized tree path;
4. only if that path reaches a leaf, applies the LM head to that one leaf and
   samples its correction token.

The committed tree prefix and correction token are then handled by the same
cache-compaction path as DDTree.

## Exact-law argument

Sampling unused internal rows early is harmless because DDTree's per-node
Target draws are independent conditional on their verified prefixes.  The new
method uses exactly the corresponding sampled row whenever the walk visits an
internal node and discards all other internal draws.  If the walk reaches a
leaf, its Target draw is generated on demand from the same leaf distribution.
Thus every root-to-first-exit event has probability

```text
product(Target probability of each accepted tree edge)
* Target probability of the final non-child token.
```

This is the DDTree ancestral output law.  The exhaustive test enumerates the
complete event distribution for a branching tree rather than checking only
matched random seeds.

## Strict work inequality

Write `L` for the indicator that the terminal node is a leaf.  DDTree projects
and samples `N` vocabulary rows.  The lazy architecture projects and samples

```text
R_lazy = I + L
```

rows.  Therefore

```text
E[R_lazy] = I + P(terminal is a leaf) <= I + 1.
```

For every tree with at least two leaves (`F >= 2`):

```text
E[R_lazy] <= I + 1 < I + F = N = R_ddtree.
```

It saves at least `F - 1` full-vocabulary rows in every round, and may save
`F` rows when the walk exits at an internal node.  If vocabulary projection
cost is `Theta(HV)` per row and probability construction/sampling is
`Theta(V)`, the affected work changes from

```text
DDTree: Theta((I + F) * (H * V + V))
Lazy:   Theta((I + P(leaf)) * (H * V + V)).
```

The ideal row-work reduction factor is

```text
(I + F) / (I + P(terminal is a leaf)).
```

The degenerate single-leaf chain has no strict projection advantage and must
not be counted as a theoretical win.

## Performance gate

Every generated round records:

- `vocabulary_projection_rows` for the lazy method;
- `full_vocabulary_projection_rows` for DDTree-equivalent eager projection;
- the internal and on-demand leaf components separately.

A valid GPU report must first confirm identical tree/model controls and
`vocabulary_projection_rows < full_vocabulary_projection_rows` on the measured
multi-leaf trees.  Throughput is then compared in one randomized, interleaved
process.  The complexity result survives a wall-clock loss; a measured speed
claim does not.
