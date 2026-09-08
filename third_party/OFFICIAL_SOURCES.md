# Official reference implementations

The directories below are unmodified shallow checkouts of the authors' repositories:

| Method | Official repository | Pinned checkout |
|---|---|---|
| DFlash | <https://github.com/z-lab/dflash> | `07ebd93db9f472af339b644bb70221ad8428328a` |
| DDTree | <https://github.com/liranringel/ddtree> | `c96427a185677bf4133ed865dd1626a5041aef9b` |

The integration runners live in the project's `scripts/` directory so that the upstream
source trees remain auditable. Do not use `git pull` when reproducing a comparison; update
the recorded commit deliberately and rerun the complete benchmark instead.
