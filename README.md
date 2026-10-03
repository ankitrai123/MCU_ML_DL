# EdgeForge

EdgeForge turns a trained ML model and a target microcontroller into
ready-to-compile embedded C (`model.h` / `model.c` / `main.c`), and — when the
board's toolchain happens to be installed — a compiled firmware image
(`.hex`/`.bin`), with a golden-vector check gating the whole thing against
the original model.

This is Phase 1: the conversion engine (library + CLI) for native/bare-metal
microcontroller targets, plus an initial Arduino CLI backend (compile-only —
see [Arduino boards](#arduino-boards-arduino-cli)) for boards whose own core
owns startup/linking, and two initial FPGA backends, each its own
deliberately separate command: `convert-fpga` (see [FPGA
backend](#fpga-backend-convert-fpga-experimental)), wrapping hls4ml for
Xilinx/Vivado HLS, and `convert-verilog` (see [Verilog/Lattice
backend](#veriloglattice-backend-convert-verilog-experimental)), a
from-scratch RTL generator for the open-source Lattice (Yosys/NextPNR) flow.
A full hosted web front end is a later phase — see [Roadmap](#roadmap).

> **Non-technical overview:** a two-page, diagram-first summary for managers is in
> [docs/EdgeForge_Overview.pdf](docs/EdgeForge_Overview.pdf) (HTML version:
> [docs/EdgeForge_Overview.html](docs/EdgeForge_Overview.html)).

## Getting started — the easy way (no coding experience needed)

This uses the point-and-click web page, not the command-line tool. You'll
need a trained model file someone gave you (a `.pkl`, `.h5`, `.keras`, or
`.onnx` file — this tool doesn't train models itself, it converts an
already-trained one). **Don't have one yet? See
[TRAINING_A_MODEL.md](./TRAINING_A_MODEL.md)** for a from-scratch,
no-ML-background walkthrough of training one in Python, using what data,
and picking what settings — then come back here. Five steps, all one-time
except the last:

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
  arduino_nano33_ble_sense_rev2 Arduino Nano 33 BLE Sense Rev2 (nRF52840, 1MB flash / 256KB RAM)
                       tier=deep       arch=arm-cortex-m4    flash=1048576B ram=262144B compiler=arduino-cli
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

### Arduino boards (arduino-cli)

`arduino_nano33_ble_sense_rev2` works differently from every board above: instead of a raw
cross-compiler invocation, EdgeForge generates an actual **Arduino sketch** (a `.ino` +
`model.h`/`model.c` in their own folder, named `<model-name>_sketch/`) and drives
[`arduino-cli`](https://arduino.github.io/arduino-cli/latest/) to compile it against that
board's own core — the same core the Arduino IDE itself uses, so it owns startup code and
linking instead of EdgeForge's generic linker script/startup stub.

Install `arduino-cli` (see its [installation
docs](https://arduino.github.io/arduino-cli/latest/installation/)), then install this
board's core once:

```bash
arduino-cli core install arduino:mbed_nano
```

`python -m edgeforge convert --board arduino_nano33_ble_sense_rev2 ...` then compiles the
generated sketch the same way every other board's `convert` compiles firmware — printing a
clear "toolchain not found" message with that same install command if `arduino-cli` isn't on
your PATH yet, rather than failing outright.

**EdgeForge compiles the sketch; it does not upload it.** Once `convert` succeeds, flash the
board yourself, either with the Arduino IDE (open the generated `.ino`, pick the board from
the Boards Manager, click Upload) or with `arduino-cli` directly:

```bash
arduino-cli upload --fqbn arduino:mbed_nano:nano33ble --port <your-port> <out-dir>/model_sketch
```

(`arduino-cli board list` shows which `--port` your board enumerated as once it's plugged in
over USB.)

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

The same UI also has a **Train a model** page (`/train`), the first slice of
the hosted-web-front-end roadmap item below: upload a CSV, pick which
column to predict and a model type (the same automated version of
[Training a model](TRAINING_A_MODEL.md)'s worked example that
`edgeforge.train.train_from_csv()` implements), and it trains, scores on a
held-out split, and surfaces beginner-friendly warnings (too few rows, a
label with almost no examples) before you ever get to inspect a confusing
result. The trained `.pkl` downloads directly, or a second form on the same
results page chains straight into the existing convert flow (pick a board,
reuse the same footprint/build/golden-vector report) without re-uploading
anything. Model types are deliberately the five estimators
`sklearn_ingest.py` actually supports (see below) — not every model type
[Training a model](TRAINING_A_MODEL.md)'s prose mentions in passing, so
nothing trained here can fail to ingest afterward.

That chained convert form is also a **target picker**: since the model
already exists server-side at that point (unlike the plain upload form
above, where a board is chosen in the same submit that provides the model),
every registered board is footprint-checked against it upfront via
`edgeforge.quantize.footprint.check_budget()`, and listed fits-first with a
fits/too-large badge and the estimated flash/RAM behind it (or the
tier-mismatch/footprint-exceeded reason when it doesn't) — see
`_board_fit()` in `webui/app.py`. It's informational, not a hard gate: the
dropdown still lists every board, and picking one that doesn't fit still
goes through `run_conversion()` and reports the same footprint error the
plain convert form would, so no gating logic is duplicated.

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
  output for a Cortex-M4 and an 8-bit MCS-51 core. A board whose
  `toolchain.kind` is `arduino-cli` renders `sketch.ino.j2` (`setup()`/`loop()`)
  instead of `main.c.j2`/a linker script/a startup stub — `model.h.j2`/`model.c.j2`
  render completely unchanged either way (an `extern "C"` guard in `model.h.j2`
  is what keeps them safe to `#include` from a C++-compiled `.ino`).

- **`edgeforge/build/toolchain.py`** — invokes the board's declared compiler
  (best-effort: reports exactly which toolchain is missing and how to
  install it if it isn't found) and, separately, the always-available host
  compiler for the golden-vector harness. A `toolchain.kind: arduino-cli`
  board dispatches to `arduino-cli compile --fqbn ... --build-path ...`
  against the generated sketch directory instead — a structurally different
  invocation (a sketch directory + a board id, not a compiler + flag list),
  reusing the same missing-toolchain/`BuildResult` shape as every other board.

- **`edgeforge/pipeline.py`** — orchestrates ingest → footprint → codegen →
  build → validate as one call returning a structured result (never raising
  on an expected `EdgeForgeError`, just recording which stage failed and
  why); `cli.py`'s `convert` command and `webui/app.py`'s `/convert` route
  both call this instead of duplicating the orchestration.

- **`edgeforge/webui/`** — the basic local web UI described above: a Flask
  app calling straight into `pipeline.run_conversion()`, plus the templates
  and a little CSS. No business logic lives here. Its `/train` routes are
  the same pattern applied to `edgeforge/train.py`'s `train_from_csv()`: the
  Flask layer only handles the upload/form/redirect plumbing, and a trained
  run's `model.pkl` sits in that run's own directory so the existing
  `/runs/<id>/download/...` route and a `/train/<id>/convert` route (which
  just calls `run_conversion()` again) both work on it unchanged.

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
| `toolchain.kind` | `raw` (default: the direct cross-compiler invocation above) or `arduino-cli` — see [Arduino boards](#arduino-boards-arduino-cli). An `arduino-cli` board sets `toolchain.fqbn` instead of `compile_args`/`link_args`/`linker_script`/`objcopy`, which go unused for it. |

See any of the five shipped board files for a complete, commented example;
`tests/test_add_board_no_code_changes.py` defines a brand-new (`raw`-kind) board purely as
YAML in a throwaway test directory and drives it through the full
ingest → footprint → codegen → build → validate pipeline as the regression
test for this claim.

## FPGA backend (`convert-fpga`, experimental)

Every board and command above shares one pipeline — `ingest → footprint →
codegen → build → validate` (see [Architecture](#architecture)) — driven by
`edgeforge convert --board <id>`. The FPGA backend deliberately isn't part
of that: there's no `boards/*.yaml` entry, no `--board` flag, and
`edgeforge/ir.py`/`edgeforge/codegen/` are never involved. Instead,
`convert-fpga` is its own command, a thin wrapper around
[hls4ml](https://fastmachinelearning.org/hls4ml/), which does its own,
independent conversion straight from a Keras model into synthesizable HLS
C++ — a fundamentally different kind of output (a hardware description, not
a CPU program) from anything `convert` generates.

```bash
pip install hls4ml tensorflow-cpu   # or: pip install -e .[fpga]
python -m edgeforge convert-fpga \
    --model examples/trained_models/iris_mlp.keras \
    --out ./build/iris_mlp_hls \
    --sample-range 0,1 --tolerance 0.2
```

(`--tolerance 0.2` here, looser than the `0.05` default — see why below.)

This always does two things, and stops there unless you ask for more:

1. **Generates the HLS project** (`hmodel.write()`) — a folder of HLS C++
   ready to open in Vivado HLS/Vitis HLS, targeting a Zynq-7020 part by
   default (`--part`, e.g. for a PYNQ-Z2/Zybo Z7-20).
2. **Validates it on the host**, with no FPGA toolchain involved: hls4ml
   compiles its own C-simulation with a plain `g++` and runs it against N
   random input samples, comparing the result against the original Keras
   model's own predictions. This is the same guarantee golden-vector
   validation gives every MCU board — proof the generated output actually
   matches the source model — aimed at hls4ml's csim instead of a
   cross-compiled `model.c`. `convert-fpga` reports SUCCESS/FAILED from this
   check alone; nothing here ever touches real hardware.

`--backend` isn't limited to the default Vivado — **Vitis, Quartus, and
Catapult are also confirmed working** (both project generation and host
C-simulation are backend-agnostic in hls4ml: every backend's csim is still
just a plain g++-compiled shared library). Pass `--synthesize` to also
attempt **real synthesis** (`hmodel.build()`) for actual LUT/FF/DSP/BRAM
numbers — each backend shells out to its own real tool (`vivado_hls` for
Vivado, `vitis-run` for Vitis, `i++` for Quartus, `catapult` for Catapult;
confirmed by reading each backend's own source, not assumed from Vivado's),
and a missing one prints that backend's own install hint — exactly like a
missing cross-compiler for any other board — and reports the HLS project as
generated-and-host-verified but not synthesized, rather than failing the
whole command or raising a confusing `TypeError` from passing one backend's
`build()` arguments to another's differently-shaped one.

**A `Softmax` output needs a looser `--tolerance` than the default.**
`--tolerance` (default `0.05`) is the max absolute difference allowed
against the original Keras model's own output, checked at this backend's
default `--precision fixed<16,6>`. Run the walkthrough above *without* the
`--tolerance 0.2` override and, on one real run, it genuinely fails:

```
HLS C-simulation validation: 14/20 samples within tolerance (0.05)
  sample 8: max abs diff 0.07568 exceeds tolerance
  sample 11: max abs diff 0.07224 exceeds tolerance
  ...
convert-fpga: FAILED (C-simulation diverged from the original model)
```

The cause is specifically `Softmax`, not general fixed-point rounding —
confirmed by elimination, not assumed: a same-sized model with a plain
linear (regression) output instead of `Softmax` held comfortably under the
`0.05` default across several retrains (worst case observed: ~0.014), while
a `Softmax`-output model like this one did not (worst case observed:
~0.13) — including when inputs were drawn from the exact `[0,1]`
distribution it trained on, ruling out an input-range mismatch as the
cause, and including at `fixed<20,10>`/`fixed<24,10>` (more bits than the
default), ruling out precision width too. hls4ml's `Softmax` uses a
table-based approximation whose own resolution is the real bottleneck. In
practice this matters less than it sounds: a few percent of absolute error
in a softmax probability essentially never changes which class scores
highest, so a looser `--tolerance` (as used above) is a reasonable choice
for a classification model — this backend just has no way to know that on
its own, since (unlike `edgeforge inspect`/`convert`) it never inspects the
model beyond its input shape and parameter count, so it always compares raw
output values rather than predicted classes.

**Keras models only — sklearn and ONNX were both tried and found not to
fit, for two different reasons, not just left undone:**

- **sklearn**: hls4ml's core has no scikit-learn converter at all (no
  `convert_from_sklearn_model`, confirmed against the installed package) —
  it targets neural-network layers, not decision trees or a bare
  `LogisticRegression`. A tree ensemble could in principle go through the
  separate [conifer](https://github.com/ssummers/conifer) project instead,
  but that's a different tool with its own integration, not something
  `convert-fpga` does. sklearn models are exactly what
  [`convert-verilog`'s classical tier](#veriloglattice-backend-convert-verilog-experimental)
  is for instead.
- **ONNX**: hls4ml does have a native `convert_from_onnx_model`, but its
  ONNX frontend turned out too narrow for the models that actually produce
  ONNX files, tried three ways: a `Gemm`-based graph (`Gemm` itself isn't in
  hls4ml's supported ONNX op list — only `MatMul`+`Add`), a hand-built
  `MatMul`+`Add` graph (crashed inside hls4ml's own parser over missing
  shape metadata on the weight initializers), and a real `skl2onnx`-exported
  model (failed on the classifier post-processing ops — `Cast`, `ZipMap` —
  every `skl2onnx` export wraps around the core network). Making this work
  would mean EdgeForge shipping its own ONNX-graph-simplification pass
  ahead of hls4ml's parser, a materially bigger undertaking than "pass the
  file through" — not attempted here.

Also not done (see [Roadmap](#roadmap) for Phase 3's status): FINN
integration (a separate framework, for binary/extreme quantization —
its own installation and typically a Brevitas-quantized PyTorch model as
input, not something layered in alongside everything above without its own
scoping pass) or producing a bitstream/programming a real board — synthesis
stops at a resource report.

## Verilog/Lattice backend (`convert-verilog`, experimental)

A second, unrelated FPGA path. hls4ml's backends are Vivado/Vitis/Quartus/
Catapult — there's no "hls4ml for Lattice" to wrap the way `convert-fpga`
wraps hls4ml for Xilinx, and the open-source Lattice flow (Yosys + NextPNR +
Project Trellis) takes synthesizable Verilog/VHDL directly, not HLS C++. So
`convert-verilog` is a from-scratch RTL generator: it lowers a model into
actual Verilog itself, reusing the *same* sklearn/Keras ingest paths and
`ModelIR` every other command uses, through entirely new templates
(`edgeforge/verilog/`) and a new numeric representation, since a small FPGA
has no float unit. Also its own command — no `boards/*.yaml`, no `--board`.

```bash
apt-get install iverilog   # host-only simulation, always required (see below for --synthesize)
python -m edgeforge convert-verilog \
    --model examples/trained_models/iris_tree.pkl \
    --out ./build/iris_tree_verilog --samples 20
# model: 75 parameters
# Verilog + testbench generated: ./build/iris_tree_verilog
# == Icarus Verilog simulation (20 samples) ==
# Verilog simulation (iverilog): 20/20 samples matched
# convert-verilog: SUCCESS
```

Every op in `edgeforge/ir.py` is implemented, in two numeric domains:

- **Classical tier** (`tree`/`linear`/`affine`/`activation`, i.e. sklearn
  models, bare or scaler-folded) runs in Q16.16 fixed point
  (`edgeforge/verilog/fixedpoint.py`): one shared signed 32-bit format, 16
  fractional bits, for every value — no per-tensor scale bookkeeping needed.
- **Deep tier** (`conv2d`/`depthwise_conv2d`/`maxpool2d`/int8 `linear`, i.e.
  Keras/TFLite models) runs in the model's own int8/int32 TFLite
  quantization, requantized between layers with an integer
  multiply-then-shift (`edgeforge/verilog/quant_math.py`) — the classic
  technique real quantized-NN hardware uses (and what TFLite Micro itself
  falls back to without a float unit), since `convert-fpga`'s "the C codegen
  can just use a plain float multiplier" shortcut isn't available here.

Every node becomes its own small state machine (`IDLE` → run → `DONE`),
chained by wiring node *k*'s `done` to node *k+1*'s `start` — one clock
cycle per loop iteration (a MAC, a pooling comparison, ...), not pipelined.
That's a deliberate priority: get an initial version that's provably
correct, even at the cost of the cycle count a real deployment would want
to optimize later.

```bash
python -m edgeforge convert-verilog \
    --model examples/trained_models/iris_mlp.keras \
    --out ./build/iris_mlp_verilog --samples 20
# model: 67 parameters
# Verilog simulation (iverilog): 20/20 samples matched
# convert-verilog: SUCCESS
```

Pass `--synthesize` for real **Yosys → NextPNR-ECP5 → ecppack** synthesis
(`--device`/`--package`/`--freq` select the target; default `45k`/
`CABGA381`/12 MHz), gated on that toolchain being installed
(`apt-get install yosys nextpnr-ecp5 fpga-trellis fpga-trellis-database` on
Debian/Ubuntu) — missing it degrades the same way a missing cross-compiler
does for any other board: source generated and host-verified, synthesis
skipped, a clear install hint printed. When it *is* installed, the numbers
are real: the tree example above synthesizes to 466 LUTs / 237 FFs / 0 DSPs
and closes timing at ~62–73 MHz against a 12 MHz target on an LFE5U-45F.

**Two honest limitations, both found by actually running the synthesis
step, not by inspection:**

1. **Every port is a flattened raw bus, with no pin-constraint (`.lpf`)
   file** — real for correctness/timing/resource numbers, but every signal
   lands on whatever pin NextPNR finds free. That's fine for "does this fit
   and time-close"; a real deployment supplies its own `.lpf` and wires this
   module to on-chip logic (UART/SPI/BRAM) rather than driving chip pins
   directly.
2. **A model with enough total input+output elements can need more I/O pins
   than the target package has, and `--synthesize`'s place-and-route step
   then fails outright** — not a resource problem (the actual compute logic
   for the CNN example in [Examples](#examples) uses under 20% of an
   LFE5U-45F's LUTs and 29% of its DSPs) but a pin-count one: that CNN's
   flattened 8×8×1 input plus its output needs over 2000 I/O pins, and even
   this backend's default package has 245. `convert-verilog` without
   `--synthesize` is unaffected (host simulation never touches pins) — this
   only blocks the optional real place-and-route step, and reports as a
   clear, handled failure (`synth_ok=False`), not a crash.

A deep-tier *regression* model's raw output — not just its predicted class
— is dequantized correctly for the host-side comparison too, via its own
tensor's TFLite scale/zero-point rather than the classical tier's Q16.16;
confirmed against a real `Dense(relu)→Dense(linear)` regression model
matching the TFLite Interpreter to ~1e-6.

Also not done yet: VHDL output (Verilog only — Yosys reads it natively,
VHDL needs the separate GHDL-Yosys plugin this doesn't set up); a real
board's pin-constraint (`.lpf`) file (there's no `--lpf` passthrough yet,
so every port lands on whatever pin NextPNR finds free — see the honest
limitations above); and a streaming or memory-mapped I/O redesign that
would let a bigger model's compute (which already fits comfortably) actually
place-and-route on a small device.

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
- **FPGA (`convert-fpga`)**: Keras models only — sklearn (no native hls4ml
  converter) and ONNX (hls4ml's own ONNX frontend too narrow for real
  ONNX exports, tried three ways) were both evaluated and ruled out, not
  just left undone; host C-simulation only (no bitstream/board
  programming) — see [FPGA backend](#fpga-backend-convert-fpga-experimental)
  for why a `Softmax`-output model needs a looser `--tolerance` than the
  default, and for exactly why sklearn/ONNX don't fit.
- **Verilog/Lattice (`convert-verilog`)**: every IR op is implemented, but a
  model with enough total input+output elements can need more I/O pins than
  the target package has, failing `--synthesize`'s place-and-route step
  outright (not a correctness problem — see [Verilog/Lattice
  backend](#veriloglattice-backend-convert-verilog-experimental)); Verilog
  output only, no VHDL; no board-specific pin-constraint (`.lpf`) file.
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
  `model.c`/`model.h`. The generated Arduino sketch (`arduino_nano33_ble_sense_rev2`)
  is the same idea via `setup()`/`loop()` and `Serial.print()` instead.
- **`arduino_nano33_ble_sense_rev2`'s `memory.flash_reserved_bytes`/`ram_reserved_bytes`**
  are estimates for the Arduino Mbed OS core's own footprint, not measured against a real
  `arduino-cli` build (unlike every other board's reserved headroom, which is a small,
  well-understood vector-table-plus-stack allowance) — tighten them once you've compiled a
  real sketch and checked its actual reported size.
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
| `train_room_comfort.py` | `Pipeline(StandardScaler, LogisticRegression)` on a small CSV of temperature/humidity readings (`room_comfort.csv`) | `arduino_nano33_ble_sense_rev2` — the worked, step-by-step example in [TRAINING_A_MODEL.md](./TRAINING_A_MODEL.md), for anyone who doesn't have a trained model yet |
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
rejection on the download route, plus the `/train` upload/configure/run
flow, its chained conversion, and the target picker's `_board_fit()`
(fitting and footprint-exceeded boards, sort order), and skips cleanly if
Flask isn't installed. `test_train.py` covers `train_from_csv()` itself for
all five supported model types, each round-tripped through
`sklearn_ingest.ingest()`.

## Roadmap (context only — not built in this phase)

- **Phase 2**: the Arduino CLI backend is partially built — `arduino_nano33_ble_sense_rev2`
  generates a sketch and drives `arduino-cli compile` (see
  [Arduino boards](#arduino-boards-arduino-cli)). Not yet done: packaging as an installable
  library folder (vs. a flat sketch), and driving `arduino-cli upload` (device programming —
  including upload port autodetection — has no precedent anywhere else in EdgeForge, so it's
  deliberately left as a manual `arduino-cli upload`/Arduino IDE step for now).
- **Phase 3**: two initial FPGA backends. `convert-fpga` wraps hls4ml to
  turn a Keras model into an HLS project and host-verify it via
  C-simulation, with best-effort real synthesis on any of Vivado/Vitis/
  Quartus/Catapult, each backend's own toolchain detected correctly (see
  [FPGA backend](#fpga-backend-convert-fpga-experimental)). Not yet done
  there: sklearn ingest (hls4ml has no scikit-learn converter at all — use
  `convert-verilog`'s classical tier instead) and ONNX ingest (hls4ml's own
  ONNX frontend was tried and found too narrow for real ONNX exports —
  both evaluated and ruled out, not just left undone), FINN integration
  (binary/extreme quantization, a separate framework with its own
  installation), and anything past a synthesis report. `convert-verilog`
  generates Verilog directly for the open-source Lattice ECP5 flow and
  covers every IR op including deep-tier regression output dequantization
  (see [Verilog/Lattice
  backend](#veriloglattice-backend-convert-verilog-experimental)). Not yet
  done there: a real board's pin-constraint (`.lpf`) file (every port is a
  raw flattened bus today), an I/O architecture that scales past a small
  device's pin count for a larger model, VHDL output, and any pipelining
  (one clock cycle per loop iteration today, correctness-first).
- **Phase 4**: a full hosted web front end (accounts, an upload *service*,
  training, a target picker) on top of this library. The `serve` command
  above is a basic single-user local UI added ahead of that — a thin layer
  over the same library, not the Phase 4 service. Training in the browser
  (upload a CSV, pick a label column and model type, train, download or
  chain into convert — see [Web UI](#web-ui) and `edgeforge/train.py`) is
  now built there too, still on the Flask dev server with no accounts. The
  post-training convert form is also a basic target picker (fits/too-large
  per board, ranked fits-first — see [Web UI](#web-ui)); it only covers the
  train-then-convert path, since checking fit for an arbitrary uploaded
  model on the plain convert form would need ingesting it before a board is
  even chosen, a bigger flow change not done here. Not yet done: accounts/
  multi-tenant isolation, an upload *service* (vs. a local run directory),
  and any real hosting/deployment mechanics (gunicorn, Docker, TLS) — all
  deliberately deferred to a later step.
