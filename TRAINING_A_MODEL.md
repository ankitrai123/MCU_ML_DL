# Training a model in Python — a guide for beginners

This is for anyone who wants to use EdgeForge but doesn't have a trained
model yet, has never trained one, and doesn't have a machine-learning
background. It covers: what data you need, how to actually train a small
model in Python, what those "parameters" everyone talks about actually are,
and what to do once you have a model — right up to handing it to EdgeForge.

It does **not** cover deep learning theory, and it deliberately sticks to
the simplest tools that work well on a tiny microcontroller. If you already
know scikit-learn, skip ahead to [the worked example](#the-worked-example-a-complete-script)
or straight to the [main README](./README.md).

**Prefer a form over a script?** `python -m edgeforge serve`'s **Train a
model** page (`/train`) walks through this exact same workflow in a
browser: upload a CSV, pick which column to predict and a model type, and
it trains, scores, and warns you about the same beginner mistakes covered
below — no code required. See [Web UI](./README.md#web-ui). The rest of
this guide explains what that page (and the worked example script) is
doing under the hood, which is worth reading once even if you use the form.

## What "training a model" actually means

You show a computer a pile of examples where you already know the right
answer, and it works out a rule that guesses the right answer for *new*
examples it's never seen. That's it — no magic.

Each example has two parts:

- **Features** — the numbers you measure. For a sensor-based project, this
  is usually one or more sensor readings (temperature, acceleration,
  light level, ...).
- **Label** — the correct answer for that example, that you already know
  (because you measured it, categorized it by hand, or looked it up).

"Training" means: give the computer many (features, label) pairs, and it
finds a rule connecting them. Afterwards, you give it *only* the features
for a brand-new situation, and it predicts the label.

## The worked example: a complete script

This guide walks through one real, complete example that's checked into
this repo and that you can run right now:

- **The data:** [`examples/room_comfort.csv`](./examples/room_comfort.csv) —
  200 made-up (but realistic) readings of room temperature (°C) and
  humidity (%), each labeled `cold`, `comfortable`, or `hot`.
- **The script:** [`examples/train_room_comfort.py`](./examples/train_room_comfort.py) —
  loads that CSV, trains a model, checks how good it is, and saves it.

Everything below explains *why* this script is written the way it is, so
you can adapt it to your own sensor and your own labels.

## Step 1: Install what you need

If you haven't already, follow **step 1** of the [main README's Getting
started section](./README.md#getting-started--the-easy-way-no-coding-experience-needed)
to install Python. Then, in a terminal opened in this project's folder:

```bash
pip install -r requirements.txt
```

(Mac: use `pip3`.) This installs everything used below, including
[pandas](https://pandas.pydata.org/), the library this guide uses to read
CSV files.

## Step 2: What data do you need?

### The shape of the data

Your data is a table: **one row per example, one column per feature, and
one column for the label.** Here's what the first few rows of
`room_comfort.csv` look like:

| temperature_c | humidity_pct | comfort     |
|--------------:|-------------:|-------------|
| 25.6          | 82.8         | comfortable |
| 15.6          | 41.0         | cold        |
| 10.1          | 77.5         | cold        |
| 30.0          | 45.9         | hot         |

Two features (`temperature_c`, `humidity_pct`), one label (`comfort`). Your
own project might have one feature (a single sensor) or several — the
approach is identical either way.

### Where this data comes from

You have two realistic options:

1. **Collect it from the actual sensor and board you'll deploy to.** This
   is the best option, and matters more than almost anything else in this
   guide: a model only predicts well on data that looks like what it was
   trained on. If you train on data from a different sensor, a lab
   bench-top setup, or someone else's dataset, and then deploy to your own
   real sensor in the real world, don't be surprised if it performs worse
   than expected — see
   [Handling a different sensor or train/deploy data mismatches](./README.md#handling-a-different-sensor-or-traindeploy-data-mismatches)
   in the main README for exactly why, and how EdgeForge helps with part
   of it.
2. **Start with a small hand-made or public dataset** (like this guide's
   `room_comfort.csv`) just to learn the *process* end-to-end, then redo it
   with real data from your own hardware before you trust the result.

### How much data, and how "clean"

There's no universal number, but for a small classifier like the ones
EdgeForge targets:

- Aim for **at least a few dozen examples of each label**, ideally more —
  `room_comfort.csv` has 200 rows split roughly evenly across its three
  labels (60/85/55). Ten examples of "hot" and 190 of "cold" will teach the
  model to just guess "cold" most of the time.
- **Consistent units.** Don't mix °C and °F, or a fraction (0–1) and a
  percentage (0–100), in the same column.
- **No blank cells.** Every row needs every column filled in.
- More *variety* (different times of day, different conditions) usually
  beats more *rows* of near-identical data.

### Making your own CSV

The easiest way: build the table in a spreadsheet (Excel, Google Sheets,
LibreOffice), one column per feature plus a label column, then use **File →
Save As / Download → CSV**. A plain text editor works too — a CSV file is
just plain text, comma-separated, exactly like the table above:

```
temperature_c,humidity_pct,comfort
25.6,82.8,comfortable
15.6,41.0,cold
```

## Step 3: Run the worked example

From this project's folder:

```bash
python examples/train_room_comfort.py
```

You should see something like:

```
accuracy on held-out test data: 85%
wrote examples/trained_models/room_comfort_model.pkl
```

That `.pkl` file is your trained model — the same kind of file the main
README's Getting Started section asks you to upload. You could stop here
and [convert it with EdgeForge right now](#step-6-hand-it-to-edgeforge).
The rest of this guide explains what that script actually did, so you can
write your own version for your own data.

## Step 4: What the script does, piece by piece

Open [`examples/train_room_comfort.py`](./examples/train_room_comfort.py)
alongside this section.

### Load the data and separate features from the label

```python
data = pd.read_csv(args.csv)
X = data[["temperature_c", "humidity_pct"]].to_numpy()
y = data["comfort"].to_numpy()
```

By convention, the features are called `X` (a table) and the label is
called `y` (one answer per row). If your data has more feature columns,
just list them all: `data[["col_a", "col_b", "col_c"]]`.

### Hold some data back for an honest test

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

This is one of the most important habits in this whole guide. `test_size=0.2`
sets aside 20% of the rows — the model never sees them during training —
so that afterwards you can check its accuracy on examples it genuinely
hasn't memorized. **Never trust an accuracy number measured on the same
data the model was trained on** — it will look great and tell you nothing
about how the model handles new situations.

### Choose a model type — the main "parameter" decision

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression()),
])
```

This is two decisions in one line, and both matter:

**1. What kind of model (`LogisticRegression()` here).** EdgeForge's
easiest path (scikit-learn) supports three kinds, all tiny enough for even
small microcontrollers:

| Model | Good for | Notes |
|---|---|---|
| `LogisticRegression()` | "Which category?" problems, like this example | Simple, fast, a solid default first try |
| `DecisionTreeClassifier()` | Same, especially when the rule is naturally "if X > some value, then..." | Very interpretable, tiniest footprint — the only option that fits the smallest 8-bit boards |
| `MLPClassifier(hidden_layer_sizes=(8,))` | Patterns too complex for the two above | A small neural network — still supported, but start simpler first |

For predicting a *number* instead of a category (e.g. an exact temperature
rather than "hot/cold"), swap in `DecisionTreeRegressor()` or
`MLPRegressor(...)` instead — this is called *regression* rather than
*classification*. (Plain `LinearRegression()` is a common regression
choice in scikit-learn generally, but EdgeForge's ingest doesn't support it
specifically — stick to the two listed here, or `LogisticRegression`/
`DecisionTreeClassifier`/`MLPClassifier` for classification, so your
`.pkl` is guaranteed to convert.)

**If you're not sure which to pick, start with `LogisticRegression()` or
`DecisionTreeClassifier()`.** They train in a fraction of a second, and
"try it and check the accuracy" is a perfectly good way to compare them —
there's no need to know the underlying math first.

**2. The scaler (`StandardScaler()` here).** Your raw features are often on
very different scales — this example's temperature ranges roughly 10–35
while humidity ranges 20–90. Several model types (including
`LogisticRegression` and `MLPClassifier`) work meaningfully better, or
train more reliably, when every feature is rescaled to a similar range
first. `StandardScaler` does this automatically by learning each feature's
typical range from your training data.

Wrapping the scaler and the model together in one `Pipeline`, exactly as
above, is important for another reason: **it's what lets EdgeForge fold
the rescaling math into the generated C code automatically**, so a raw
sensor reading on the real board gets normalized on-device exactly the way
your training data was — see
[Handling a different sensor](./README.md#handling-a-different-sensor-or-traindeploy-data-mismatches)
for the full story. Always pickle the whole `Pipeline`, never just the
`LogisticRegression()`/`DecisionTreeClassifier()` step by itself.

### Train it

```python
model.fit(X_train, y_train)
```

This is the actual "training" step — everything before it was preparation.

### Check how good it is

```python
print(f"accuracy on held-out test data: {model.score(X_test, y_test):.0%}")
```

This example genuinely prints `85%` on its held-out test rows. What counts
as "good enough" depends entirely on your project, but as a rough sense of
scale: for a 3-way classification like this one, random guessing would get
about 33%, so 85% means the model has clearly learned something real.
70–90%+ is typical for a simple sensor-based classifier like this; if
you're seeing much less, the most common fixes are (in order of how often
they actually help): collect more/cleaner/more varied data, double-check
your labels are correct, then try a different model type.

### Save it

```python
with open(args.out, "wb") as f:
    pickle.dump(model, f)
```

This writes the trained `Pipeline` — scaler and model together — to a
`.pkl` file. This is the exact file format EdgeForge ingests.

## Step 5: Adapting this to your own data

1. Replace `room_comfort.csv` with your own CSV (or point `--csv` at it:
   `python examples/train_room_comfort.py --csv my_data.csv`).
2. Update the feature and label column names in the script to match your
   CSV's actual column headers.
3. If you're predicting a category (a label like `"hot"`/`"cold"`, not a
   number), keep `LogisticRegression`/`DecisionTreeClassifier`/`MLPClassifier`.
   If you're predicting a number, switch to the matching `*Regressor`.
4. Re-run the script, check the printed accuracy, and iterate on your data
   (not the code) if it's lower than you'd like — see the note above.

## Step 6: Hand it to EdgeForge

You now have a `.pkl` file. From here it's exactly the main README's
process:

- **Trained it in the browser?** The **Train a model** page's results
  screen has a "Convert this model for a board" form right below your
  score — pick a board there and skip straight to a converted result, no
  `.pkl` download/re-upload needed.
- **Easiest (starting from a `.pkl` file):** `python -m edgeforge serve`,
  then open `http://127.0.0.1:5000`, upload your `.pkl`, and pick a board —
  see [Getting started](./README.md#getting-started--the-easy-way-no-coding-experience-needed).
- **Command line:**
  ```bash
  python -m edgeforge convert --model examples/trained_models/room_comfort_model.pkl \
      --board arduino_nano33_ble_sense_rev2 --out ./build/room_comfort_arduino
  ```
  Running this exact command against this guide's own example produces:
  ```
  kind=classical task=classification params=13 weight_bytes=52 activation_bytes=8
    affine             scaler0  in=['input'] out=['scaler0_out']
    linear             fc0  in=['scaler0_out'] out=['output']

  golden-vector validation: 20/20 samples matched
  convert: SUCCESS
  ```
  The `affine` step is your `StandardScaler`, folded automatically into the
  generated C — you didn't have to write any rescaling code by hand. The
  "20/20 samples matched" line means EdgeForge double-checked the generated
  C against your actual Python model and they agree.

Pick whichever board matches your hardware — see `python -m edgeforge
list-boards`, or the [Examples table](./README.md#examples) for a few
starting points. `arduino_nano33_ble_sense_rev2` is a good choice if
that's the board you have; see
[Arduino boards](./README.md#arduino-boards-arduino-cli) for how to finish
compiling and uploading to it.

## Step 7: Getting it onto real hardware

Two things are still yours to do by hand, on purpose — EdgeForge generates
a *starting point*, not a finished driver:

1. **Wire up your real sensor.** The generated code has a stubbed
   `read_sensor()` function that just returns zeros — replace it with code
   that reads your actual sensor, in the **same units your training data
   used**. If your deployed sensor is physically different from whatever
   you trained on, see
   [Handling a different sensor](./README.md#handling-a-different-sensor-or-traindeploy-data-mismatches)
   — this is the single most common reason a correctly-converted model
   misbehaves on real hardware.
2. **Compile and flash it.** How, exactly, depends on your board — see
   [Toolchain setup](./README.md#toolchain-setup) (most boards) or
   [Arduino boards](./README.md#arduino-boards-arduino-cli) (Arduino Nano
   33 BLE Sense Rev2).

## Common beginner mistakes

- **Measuring accuracy on the training data instead of held-out test
  data.** It will look better than it is.
- **Skipping the scaler** when your features are on very different scales
  (like this example's temperature vs. humidity) — some models train
  poorly without it.
- **Pickling the bare model instead of the whole `Pipeline`** — you lose
  the scaler, and EdgeForge will ask you to fix this specifically (or, on
  an older version, silently expect pre-scaled input).
- **Training on one sensor, deploying to another** without recalibrating —
  see the sensor-mismatch section linked above.
- **Very few examples, or wildly imbalanced labels** (e.g. 190 of one label
  and 3 of another).

## Where to go next

- [README: Scope and known limitations](./README.md#scope-and-known-limitations) —
  exactly which scikit-learn/Keras/ONNX model shapes EdgeForge accepts.
- [README: Handling a different sensor or train/deploy data mismatches](./README.md#handling-a-different-sensor-or-traindeploy-data-mismatches) —
  the deeper dive on sensor calibration and dataset shift this guide keeps
  pointing to.
- [README: Examples](./README.md#examples) — more worked examples,
  including a small neural network (Keras) and a tiny image-style CNN.
- [README: Adding a board](./README.md#adding-a-board) — if your exact
  board isn't listed yet.
