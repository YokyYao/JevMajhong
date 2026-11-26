# Real RI class-loss curves and private journal snapshots

The active fresh RI control logs a changed max-discard class objective and
original hard-action CE separately. The existing hard-CE-only final figure
export asserts equality of those quantities and must remain unchanged.

Add a versioned exporter for actual complete-prefix snapshots and audited
finals. Plot train/validation class objective and original CE separately, RI
accept/refuse recall separately for legacy/teacher, observed complete-East
reward (raw terminal net/8,000), hand-win/top-two/first/deal-in rates. Connect
only actually observed registered points. Supervised reward stays null; no
smoothing, interpolated observations or synthetic training reward is added.
Match every process point on the same16 games/two logical seeds; final64 is
separate, using its recorded matching16 subset on the process curve.

Capture exact live journal bytes privately before reading complete lines.
Never bind a growing journal as an immutable current file. Export only minimal
training scalar columns (step, objective, original CE, learning rates, exposures),
aggregate validation confusion counts and observed aggregate game metrics to a
separate public directory. Keep full snapshots and per-update source/gradient
diagnostics outside that public directory. Record hashes and actual point
coverage. An unfinished figure must say unfinished and disclose train,
validation and game endpoints separately. A final requires actual independent
full608/57,140-exposure/final64 evidence in addition to complete scalar points.

Run unit and real export checks only in a bounded GPU scheduler allocation.
The first useful two-game-point snapshot waits for the running315768 job's
actual152 validation and complete16-game point in the allocated sidecar job,
with a finite timeout, log and exit; no login polling script or new model
inference is used. Capture actual files and inspect the rendered PNG. This
checks visualization and privacy accounting, not student improvement, parent
admission or the full four-course/v8 goal. Leave active/completed source and
data unchanged. Publish only source/plans/aggregate results and selected public
scalar/figure artifacts, never the private snapshots.
