# EdgeForge

EdgeForge turns a trained ML model and a target microcontroller into
ready-to-compile embedded C (`model.h` / `model.c` / `main.c`), and — when the
board's toolchain happens to be installed — a compiled firmware image
(`.hex`/`.bin`), with a golden-vector check gating the whole thing against
the original model.

This is Phase 1: the conversion engine (library + CLI) for native/bare-metal
microcontroller targets. A web front end, an Arduino CLI backend, and an FPGA
backend are later phases — see [Roadmap](#roadmap).

## Getting started — the easy way (no coding experience needed)

This uses the point-and-click web page, not the command-line tool. You'll
need a trained model file someone gave you (a `.pkl`, `.h5`, `.keras`, or
`.onnx` file — this tool doesn't train models itself, it converts an
already-trained one). Five steps, all one-time except the last:

**1. Install Python** (skip if you already have it). Go to
[python.org/downloads](https://www.python.org/downloads/) and download the
installer for your system.
- **Windows:** run the installer, and **check the box that says "Add
  python.exe to PATH"** on the very first screen before clicking Install —
  this is the single most common thing people miss.
- **Mac:** run the installer normally. On Mac, use `python3` and `pip3`
  (not `python`/`pip`) in the commands below.

**2. Get the project files onto your computer.** If you were given a link
to this repository, open it in a browser, click the green **Code** button,
then **Download ZIP**, and unzip it somewhere you'll remember (like your
Desktop). You should end up with a folder containing a file named
`README.md` (this file) and a folder named `edgeforge`.

**3. Open a terminal in that folder.**
- **Windows:** open the folder in File Explorer, click once in the address
  bar at the top (where the folder path is shown), type `cmd`, and press
  Enter. A black window opens — that's your terminal, already pointed at
  the right folder.
- **Mac:** open the folder in Finder, right-click inside it, and choose
  **New Terminal at Folder** (or open the Terminal app and type `cd ` then
  drag the folder into the window and press Enter).

**4. Type these two commands, pressing Enter after each one** (the first
one takes a minute or two and prints a lot of text — that's normal):

```
pip install -r requirements.txt
python -m edgeforge serve
```

Leave that window open — it's now running a small local web server just
for you, on your own computer. (Mac: use `pip3` instead of `pip` if the
first command says "command not found".)

**5. Open your web browser** and go to `http://127.0.0.1:5000`. Choose
your model file, pick a board from the dropdown (**if you're not sure which
one, pick `stm32f411`** — it accepts the widest range of models), and click
**Convert**. A green **SUCCESS** badge means it worked; the page lists the
files it generated with a download link next to each, plus a "Download all
as .zip" button.

You do **not** need to install anything else to get useful, correct C code
out of this — that's all steps 1–5 above. Installing a compiler (see
[Toolchain setup](#toolchain-setup) below) is only needed for two *extra*
things: getting a ready-to-flash firmware file instead of just source code,
and having the tool double-check its output against your original model
before you trust it. If you skip that, the results page will say so plainly
instead of failing.

When you're done, go back to the terminal window from step 4 and press
`Ctrl+C` to stop the server.

*Prefer the command line, or setting this up for a team/CI? See
[Quick start](#quick-start) below — it's the same tool, run without the
web page.*

## Why this exists, in one sentence

The same generator code produces correct C for a 512KB-flash/128KB-RAM
Cortex-M4 and a 256B-RAM 8-bit 8051 core, because everything that varies
between boards — integer widths, memory-placement qualifiers, calling
conventions, toolchain invocation — is *data* on a board profile, not a
`case` statement in the generator.

## Quick start

```bash
pip install -r requirements.txt
# Cross toolchains, Linux (Debian/Ubuntu) -- best-effort: EdgeForge still
# generates + validates source without these, see "Toolchain setup" below
# for Windows/macOS and for the host compiler validation always needs.
apt-get install gcc-arm-none-eabi sdcc

# Produce a demo model, then convert it
python examples/train_iris_tree.py
python -m edgeforge convert \
    --model examples/trained_models/iris_tree.pkl \
    --board stm32f411 --out ./build/iris_tree_stm32

python -m edgeforge convert \
    --model examples/trained_models/iris_tree.pkl \
    --board 8051_at89s52 --out ./build/iris_tree_8051
```

Each `convert` prints the model's parameter count and estimated flash/RAM
footprint, checks it against the board's declared budget *before* generating
anything, writes `model.h`/`model.c`/`main.c` (+ a linker script and startup
stub for bare-metal ARM targets), invokes the board's toolchain if it's
installed, and runs the golden-vector check (always on the host, regardless
of whether the target toolchain is available). A model that doesn't fit a
board's RAM/flash fails at the footprint step with a specific message, before
any C is written.

```
$ python -m edgeforge list-boards
boards in /path/to/boards:
  8051_at89s52         AT89S52 (8051 core, 8KB flash / 256B RAM)
                       tier=classical  arch=mcs51            flash=8192B ram=256B compiler=sdcc
  esp32_native         ESP32 (Xtensa LX6, simplified single-region memory model)
                       tier=deep       arch=xtensa-lx6       flash=1048576B ram=327680B compiler=xtensa-esp32-elf-gcc
  native_cortex_m4     Native Cortex-M4 (generic 32-bit baseline)
                       tier=deep       arch=arm-cortex-m4    flash=262144B ram=65536B compiler=arm-none-eabi-gcc
  stm32f411            STM32F411 (Cortex-M4, 512KB flash / 128KB RAM)
                       tier=deep       arch=arm-cortex-m4    flash=524288B ram=131072B compiler=arm-none-eabi-gcc
```

`python -m edgeforge inspect --model <path> [--board <id>]` ingests a model
and prints its IR (and, with `--board`, the footprint report) without
generating anything — useful for sanity-checking a model before committing to
a full convert.

### Toolchain setup

Two *separate* things need a C compiler, and it's easy to install one and
still hit an error because the other is missing:

1. **The host compiler** — validation always compiles the generated C on
   your machine and runs it there (never on real hardware), regardless of
   which board you're targeting. Without it, `convert` fails at the
   golden-vector step with a message naming the missing compiler, even if
   the target board's own toolchain is installed.
2. **The target board's cross-compiler** (`arm-none-eabi-gcc` for
   stm32f411/native_cortex_m4, `sdcc` for 8051_at89s52) — optional. Without
   it, `convert` still succeeds: it generates the C source and runs the
   golden-vector check, and just prints which toolchain is missing instead
   of producing a compiled `.hex`.

**Linux (Debian/Ubuntu):** `gcc` (host) is normally already present;
`apt-get install gcc-arm-none-eabi sdcc` gets the other two.

**macOS:** `xcode-select --install` gets a host `gcc`/`clang`;
`brew install --cask gcc-arm-embedded && brew install sdcc` gets the other two.

**Windows:** there's no built-in C compiler at all, so `apt-get` (a Linux
package manager) won't exist, and you need the host compiler explicitly.
[MSYS2](https://www.msys2.org/) provides all three in one place:

```powershell
winget install --id=MSYS2.MSYS2 -e
```

Open the **"MSYS2 MSYS"** shortcut from the Start menu and run:

```bash
pacman -Syu   # if it asks you to close and reopen the window partway through, do so, then run this again
pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-arm-none-eabi-toolchain mingw-w64-x86_64-sdcc
```

Then add `C:\msys64\mingw64\bin` to your PATH (search "Environment
Variables" in the Start menu → *Edit environment variables for your
account* → *Path* → *New*), open a **new** PowerShell/cmd window, and verify:

```powershell
gcc --version
arm-none-eabi-gcc --version
sdcc --version
```

`python -m edgeforge convert ...` (in your regular PowerShell/cmd, with
Python already installed) will then find all three.

### Web UI

For end users who'd rather not use the CLI, `python -m edgeforge serve`
starts a basic local web UI at `http://127.0.0.1:5000`: upload a model, pick
a board from a dropdown (with each board's tier/flash/RAM/toolchain shown
inline), click Convert, and get the same footprint/build/golden-vector
report as the CLI plus download links for every generated file (or a
"download all as .zip" button).

```bash
pip install -r requirements.txt   # includes flask
python -m edgeforge serve
```

It's a thin layer over the same `edgeforge.pipeline.run_conversion()` the
CLI calls — no ingest/codegen/build/validate logic is duplicated. It's meant
for a single local user on their own machine, the same trust model as the
CLI: no accounts, no auth, and the same "only convert models you trust"
caveat (ingesting a `.pkl` means unpickling it). Don't bind `--host` to a
public interface without addressing that.

## Architecture

```
model file --[ingest]--> ModelIR --[footprint]--> gate --[codegen]--> C source --[build]--> firmware
                                                                              \--[validate]--> golden-vector check
```

- **`edgeforge/boards/`** — `schema.py` defines the typed `BoardProfile`
  (memory budget, C type roles, storage qualifiers, toolchain command
  templates) and validates a board YAML on load; `registry.py` discovers
  every `boards/*.yaml` file. **Nothing outside this package ever branches on
  a board's id or architecture name** — see [Adding a board](#adding-a-board).

- **`edgeforge/ir.py`** — the intermediate representation every ingest
  backend lowers into and every codegen template renders from: `TensorSpec`
  (a weight/bias/activation buffer, with quantization params when
  applicable) and `Node` (`tree`, `linear`, `conv2d`, `depthwise_conv2d`,
  `maxpool2d`, `activation`, `affine`). A decision tree and a quantized CNN
  are both just a chain of these nodes; a `LogisticRegression` and one layer
  of a quantized TFLite MLP are literally the same `linear` op, differing
  only in the weight tensor's dtype. `affine` (`y = x*scale + shift`,
  per-feature) is how a fitted sklearn scaler gets folded into the same
  IR/codegen machinery — see the `sklearn_ingest.py` bullet below.

- **`edgeforge/ingest/`** — one module per source format:
  - `sklearn_ingest.py`: `DecisionTreeClassifier`/`Regressor`,
    `LogisticRegression`, `MLPClassifier`/`Regressor` — either bare, or
    wrapped in a two-step `Pipeline(StandardScaler|MinMaxScaler,
    <one of these>)`, in which case the fitted scaler is folded into the
    generated C as a leading `affine` node instead of being dropped. Any
    other `Pipeline` shape still needs the final estimator pickled by
    itself, with the rest of the preprocessing folded into training by hand.
  - `keras_ingest.py`: `.h5`/`.keras` → `TFLiteConverter` post-training int8
    quantization → the *TFLite flatbuffer's own schema* (not the
    `Interpreter`'s private introspection or the CPU delegate's rewritten
    graph) for `FULLY_CONNECTED`, `CONV_2D`, `DEPTHWISE_CONV_2D`,
    `MAX_POOL_2D`, `AVERAGE_POOL_2D`, `SOFTMAX`. `Flatten()`'s
    dynamic-shape op cluster (`SHAPE`/`STRIDED_SLICE`/`PACK`/`RESHAPE`) is
    transparent: a flat row-major C buffer never needs reshaping, so ingest
    walks through those ops via a tensor-alias map instead of emitting nodes
    for them.
  - `onnx_ingest.py`: plain feedforward graphs (`Gemm` or `MatMul`+`Add`
    chains, ReLU, a final Softmax/Sigmoid) — the same territory as sklearn's
    MLP, so it reuses the identical `linear`/`activation` IR ops and needs no
    new codegen at all.

  Every ingester returns an `IngestResult`: the IR, a `reference_fn` that
  calls the *real* source-framework model (not a reimplementation of the
  IR — see [Validation](#validation) for why that distinction matters), and
  an `input_sampler` for golden-vector test generation.

- **`edgeforge/quantize/footprint.py`** — parameter count and estimated
  flash (constant/weight storage) and RAM (activation scratch + input/output
  buffers) footprint, checked against the board's declared budget *before*
  any code is generated. A `model_tier: classical` board (8051) rejects a
  neural-net model here too, before wasting a single template render on it.

- **`edgeforge/codegen/`** — Jinja2 templates (`model.h.j2`, `model.c.j2`,
  `main.c.j2`, one op macro per IR node type in `ops.j2`) rendered against a
  context built purely from the IR and the board profile. The *only* thing
  that changes between an ARM board and the 8051 is which values
  `board.type_of(...)` and the storage-qualifier lookups return — literally
  the same `.j2` files, including the linker script template, render correct
  output for a Cortex-M4 and an 8-bit MCS-51 core.

- **`edgeforge/build/toolchain.py`** — invokes the board's declared compiler
  (best-effort: reports exactly which toolchain is missing and how to
  install it if it isn't found) and, separately, the always-available host
  compiler for the golden-vector harness.

- **`edgeforge/pipeline.py`** — orchestrates ingest → footprint → codegen →
  build → validate as one call returning a structured result (never raising
  on an expected `EdgeForgeError`, just recording which stage failed and
  why); `cli.py`'s `convert` command and `webui/app.py`'s `/convert` route
  both call this instead of duplicating the orchestration.

- **`edgeforge/webui/`** — the basic local web UI described above: a Flask
  app calling straight into `pipeline.run_conversion()`, plus the templates
  and a little CSS. No business logic lives here.

- **`edgeforge/validate/golden.py`** — the main defense against silent
  precision bugs. N sample inputs run through the *original* model
  (`sklearn.predict`, the real TFLite `Interpreter`, `onnxruntime`) and
  through the generated C, compiled into a **host-side** test harness (never
  the target board's cross-toolchain, and never flashed to hardware).
  Classifiers are graded on predicted-class agreement (argmax is invariant
  to any monotonic transform, which is also why softmax/sigmoid before a
  classification output is elided from codegen entirely — see below);
  regressors are graded on numeric closeness with a quantization-aware
  tolerance (2 quantization steps for a deep-tier model, a tight relative
  tolerance for classical float32 math).

## Adding a board

Drop a new `boards/<id>.yaml` file. That's it — no Python changes.
`boards/esp32_native.yaml` exists specifically to prove this: it targets a
different CPU architecture and compiler family (Xtensa/`xtensa-esp32-elf-gcc`,
vs. the ARM/`arm-none-eabi-gcc` and 8051/`sdcc` boards) and reuses the exact
same generic linker script template the ARM boards use, purely by supplying
different data.

A board file declares:

| Section | What it controls |
|---|---|
| `model_tier` | `classical` (sklearn-style models only) or `deep` (also accepts quantized neural nets) |
| `memory` | flash/RAM sizes, reserved headroom for runtime/stack, and (for non-reentrant 8-bit toolchains) a RAM overhead multiplier — see the comment in `boards/8051_at89s52.yaml` |
| `c_dialect.types` | the concrete C type for each of 8 logical roles (int8 weight, int32 bias/accumulator, the classical float type, loop-index type, ...) |
| `c_dialect.rom_qualifier` / `rom_guard_macro` | an extra storage keyword for flash-resident const data (SDCC's `__code`) and the compiler-predefined macro that guards it — empty/`null` for a target where plain `const` + the linker script already places data in flash |
| `toolchain` | compiler binary, flags, linker script template name, objcopy step (or none, for a compiler that emits the final format directly), and whether it needs a separate compile-then-link pass (SDCC can't take more than one source file per invocation) |

See any of the four shipped board files for a complete, commented example;
`tests/test_add_board_no_code_changes.py` defines a brand-new board purely as
YAML in a throwaway test directory and drives it through the full
ingest → footprint → codegen → build → validate pipeline as the regression
test for this claim.

## Scope and known limitations

- **sklearn**: a bare fitted `DecisionTree*`/`LogisticRegression`/`MLP*`
  estimator, or that same estimator as the final step of a two-step
  `Pipeline(StandardScaler|MinMaxScaler, <estimator>)` — the scaler is folded
  into the generated C (see
  [Handling a different sensor or train/deploy data mismatches](#handling-a-different-sensor-or-traindeploy-data-mismatches)).
  Any other `Pipeline` shape (more preprocessing steps, a different
  transformer such as `PCA`) isn't ingested directly — extract the final
  estimator and fold that preprocessing into training instead. MLP hidden
  activations are limited to `relu`/`identity`: `tanh`/`logistic` would need
  a math library, which EdgeForge avoids linking so even the 8051 tier stays
  freestanding.
- **Keras/TFLite**: `FULLY_CONNECTED`, `CONV_2D`, `DEPTHWISE_CONV_2D`,
  `MAX_POOL_2D`, `AVERAGE_POOL_2D`, `SOFTMAX`, plus transparent
  reshape/flatten. No LSTM/attention/BatchNorm-as-its-own-op. A single-unit
  sigmoid classification output isn't supported (argmax needs >=2 units) —
  export a 2-unit softmax layer instead.
- **ONNX**: plain feedforward graphs only (`Gemm`/`MatMul`+`Add` chains) —
  the "small MLP" tier, not arbitrary ONNX graphs. No int8 quantization path
  for ONNX in this phase (it would need ONNX→TF→TFLite, a fragile multi-hop
  conversion this phase doesn't need for MLP coverage); ONNX models render
  through the classical (float32) codegen path.
- **Deep-tier requantization** uses a plain `float` multiplier rather than
  TFLite Micro's integer-only fixed-point-multiply-and-shift trick. Every
  deep-tier board in the registry has usable float (hardware FPU or
  soft-float via libgcc/newlib) — the 8051 tier never runs this path at
  all — so this is simpler to generate and verify while landing within the
  same few-ULP tolerance a fixed-point implementation would.
- **Footprint estimation on 8051** is a heuristic, not exact: SDCC's default
  non-reentrant calling convention gives every function's locals a permanent
  static slot rather than sharing stack space, so real RAM use (and its
  fragmentation) isn't perfectly predictable from Python. The pre-flight
  check catches most cases; a rare model that passes the estimate can still
  fail at actual `sdcc` link time with a clear "reduce the model" message
  rather than silently producing broken output.
- The generated `main.c` is a wiring demo (stubbed `read_sensor()` → quantize
  → `model_infer()` → a `volatile` result a debugger can observe) — replace
  `read_sensor()` and the result-handling with real driver code; don't edit
  `model.c`/`model.h`.
- `--sample-range` values with a negative lower bound need `=`, e.g.
  `--sample-range=-2,8` — otherwise argparse mistakes `-2,8` for another
  flag.

## Handling a different sensor or train/deploy data mismatches

Two related problems show up once a converted model meets real hardware:

**1. Your training pipeline normalized its input, and the generated C didn't
know.** If you trained on `scaler.transform(X)` output (e.g. via a
`Pipeline(StandardScaler(), LogisticRegression())`) rather than raw feature
values, the model only ever saw standardized numbers — feeding it a raw
sensor reading at inference time produces meaningless predictions, even
though the model itself converted without error. **This is now handled
automatically**: pickle the whole `Pipeline` (not just the final estimator)
and EdgeForge folds the fitted `StandardScaler`/`MinMaxScaler` into the
generated C as an extra step that runs before inference, using the exact
`mean_`/`scale_` (or `min_`/`data_range_`) values learned during training —
see `examples/train_iris_logreg_scaled.py`. Feed `read_sensor()`'s raw output
straight in; the generated code does the rescaling.

**2. Your deployed sensor isn't the one you trained with.** Folding a scaler
replays the *statistical* normalization training used — it can't fix a raw
reading that means something physically different to begin with. If your
training data came from one sensor (say a 10-bit ADC, 0–1023) and the board
in the field has a different one (a 12-bit ADC, 0–4095, or a different
sensitivity/units entirely), those raw numbers aren't on the same scale at
all, scaler or no scaler. Fix this in `read_sensor()`, before any EdgeForge
code runs: convert the new sensor's raw output into the same physical units
your training sensor used. A two-point linear calibration is usually enough
— read the new sensor at two known reference points and solve for
`physical = raw * gain + offset`:

```c
/* Example: calibrated against two known reference readings. */
#define SENSOR_GAIN   0.0244f   /* (ref2_physical - ref1_physical) / (ref2_raw - ref1_raw) */
#define SENSOR_OFFSET (-1.2f)   /* ref1_physical - ref1_raw * SENSOR_GAIN */

raw_out[i] = (float)adc_read(i) * SENSOR_GAIN + SENSOR_OFFSET;
```

Once `read_sensor()` produces values in the same physical units training
data used, a folded scaler (if any) handles the rest.

**3. Your training data doesn't represent real-world deployment
conditions.** No code-gen step can fix this automatically — it's a data
problem, not a conversion problem. If accuracy on the physical board is
worse than what you saw training/testing in Python even after (1) and (2)
are ruled out, capture a batch of real on-device sensor readings and compare
their range against your training data's range (`edgeforge inspect --model
...` prints the ingested model's structure, but eyeballing `X.min(axis=0)`/
`X.max(axis=0)` on both datasets in Python is the direct check). If they
diverge, retrain (or fine-tune) on data that includes real deployment
conditions rather than trying to compensate for the gap on-device.

## Examples

`examples/` trains small demo models and saves them to
`examples/trained_models/`:

| Script | Model | Suggested board |
|---|---|---|
| `train_iris_tree.py` | `DecisionTreeClassifier` on iris | `8051_at89s52` |
| `train_iris_logreg.py` | `LogisticRegression` on iris | `stm32f411` |
| `train_iris_logreg_scaled.py` | `Pipeline(StandardScaler, LogisticRegression)` on iris | `stm32f411` (demonstrates folding a scaler into the generated C — see [Handling a different sensor or train/deploy data mismatches](#handling-a-different-sensor-or-traindeploy-data-mismatches)) |
| `train_iris_mlp.py` | `MLPClassifier` on iris | `stm32f411` (too large for `8051_at89s52`'s 256B RAM — a real, instructive rejection) |
| `train_keras_mlp.py` | small dense Keras MLP on iris | `stm32f411`, `native_cortex_m4`, or `esp32_native` (`--sample-range=-3,9`) |
| `train_keras_cnn.py` | `Conv2D`→`MaxPool2D`→`DepthwiseConv2D`→`Dense` on a synthetic "which corner is the blob in" task | `stm32f411` or `native_cortex_m4` (`--sample-range=0,1`) |

## Testing

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

The suite trains small real models (sklearn + a few short-epoch Keras/ONNX
ones) rather than relying on fixtures checked into the repo, and includes a
"deliberately corrupt a weight and confirm golden validation catches it"
meta-test, so a green suite means the validation harness is actually
checking something. Tests that invoke the cross toolchains
(`arm-none-eabi-gcc`, `sdcc`) skip cleanly if those aren't installed; golden
validation itself never needs them (host-only, per the design above).
`test_webui.py` covers the upload/convert/download flow and path-traversal
rejection on the download route, and skips cleanly if Flask isn't installed.

## Roadmap (context only — not built in this phase)

- **Phase 2**: an Arduino CLI backend — generate a sketch + library folder,
  drive `arduino-cli compile`/`upload` headlessly.
- **Phase 3**: an FPGA backend for quantization-aware models, wrapping
  hls4ml/FINN rather than writing HLS generation from scratch.
- **Phase 4**: a full hosted web front end (accounts, an upload *service*,
  training, a target picker) on top of this library. The `serve` command
  above is a basic single-user local UI added ahead of that — a thin layer
  over the same library, not the Phase 4 service.
