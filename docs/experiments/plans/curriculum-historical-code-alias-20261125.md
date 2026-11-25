# Exact historical Python source mapping

One original v8 ancestor data manifest names a project Python module whose
current bytes have since changed. Preserve that failed strict check and the
original manifest. Resolve only an explicit historical source declaration;
never treat the current file as if it had the old hash.

The resolver requires the unchanged original manifest and its exact declared
module hash, the identical package-relative Python path under an archived
project code package, and membership at that hash in a verified actual ancestor
protocol's code inputs. Reject data/weight aliases, unnecessary aliases,
changed/unbound archives, changed manifests, different module paths and
symlink redirection. The caller must independently verify actual ancestry and
producer-receipt bindings. Merely supplying a dictionary is not that proof.

Record the original key/expected hash, the observed current hash and the exact
archived path/hash separately. No source or manifest is rewritten, and no
archived code is executed. Bind the declaration, original manifest, actual
protocols, archived source and resulting evidence. The mapping is a source-role
resolution only; it cannot assert complete raw-source exclusion, admit the
future final candidate, open reserved games or establish model strength.

Apply the resolver only in allocated finite GPU-job resources. Public material
contains the resolver, rejection tests, this plan and aggregate accounting.
Original manifests, absolute-path mappings, raw records and process receipts
remain private. Existing live training inputs and weights stay unchanged.
