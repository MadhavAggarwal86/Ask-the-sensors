# Ask the Sensors

Grounded, explainable activity question answering from wearable accelerometer + gyroscope signals
(ExtraSensory, 7 classes: lying down, sitting, standing in place, standing and moving, walking,
running, bicycling).

**Time base:** every timestamp is reported in **seconds from the start of the recording**.

## Design in one picture

```
recording ──► preprocess ──► recognise ──► aggregate ──► answer
             25 Hz grid,     per-window    timeline of   parse question → run operation
             6 s windows,    features +    intervals     over the timeline → fill the
             gap mask        classifier                  6-field output with cited evidence
```

| Layer | Code | What it does |
|---|---|---|
| Preprocess | `src/ats/preprocess.py` | unit normalisation (m/s² → g), resample to 25 Hz, keep gaps as NaN, 6 s windows / 3 s hop, drop windows with <60% data |
| Features | `src/ats/features.py` | 35 named, physical features (variability, cadence, periodicity, tilt, rotation). The same numbers are quoted in explanations |
| Recognise | `src/ats/model.py` | HistGradientBoosting, class-balanced, handles missing gyro natively; trained with gyro dropout |
| Aggregate | `src/ats/timeline.py` | sticky-HMM Viterbi decoding of window probabilities, interval merging across ExtraSensory's 40 s gaps, durations/bouts/onsets |
| Answer | `src/ats/qa/` | `parser.py` question → intent; `operations.py` intent → answer + evidence; `explain.py` explanations built only from measured values; `format.py` output format |
| Evaluate | `src/ats/eval/`, `scripts/evaluate.py` | auto-generated questions from ground-truth labels, scoring rules per type, Figures 1, 3, 5 |
| Compress | `src/ats/compress.py`, `scripts/benchmark.py` | boosting-round pruning, distillation into a tree / small MLP, int8 quantisation; accuracy vs size and latency, Figure 4 |

**Grounding rule:** the language layer never produces a number or timestamp. Every value comes from
the timeline, and every explanation is a template filled with feature values of the cited windows.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Reproduce

```bash
# 1. Data (~16 GB of raw acc + gyro; labels alone are ~1 MB with --labels-only)
python scripts/download_data.py

# 2. Window features + labels for every user
python scripts/build_dataset.py --source extrasensory

# 3. Train on training users, Figure 2 (confusion matrix) on test users
python scripts/train.py --source extrasensory

# 4. (Optional) re-tune the temporal decoder on held-out TRAINING users; copy the result into config.py
python scripts/tune_smoothing.py --source extrasensory

# 5. End-to-end QA evaluation on held-out users (Figures 1, 3, 5);
#    --smoothing moving_average reproduces the first baseline
python scripts/evaluate.py --source extrasensory --robustness

# 6. Efficiency: pruned / distilled / int8 / accelerometer-only variants, Figure 4 (Pareto front)
python scripts/benchmark.py --source extrasensory
```

The user split is fixed in `configs/split.json` (created on first run) so every member gets the same numbers.

### Answer questions about a new recording

```bash
python scripts/answer.py --recording path/to/recording.csv \
    --question "How long was the user walking?" \
    --question "Did the user begin running at any point, and if so, when?"
```

The recording CSV needs a time column (`t`, `time` or `timestamp`; seconds or epoch seconds/ms) and
`acc_x, acc_y, acc_z` (+ optional `gyro_x, gyro_y, gyro_z`). Add `--questions file.txt` for one
question per line and `--json out.jsonl` for machine-readable output.

### Without the real data (pipeline check only)

```bash
python scripts/make_synthetic.py --users 10 --recordings 2 --minutes 40
python scripts/build_dataset.py --source synthetic
python scripts/train.py --source synthetic
python scripts/evaluate.py --source synthetic
pytest -q
```

Synthetic signals are crude caricatures used only to test the code. **Never report synthetic numbers as results.**

## Repository layout

```
src/ats/            library (config, preprocess, features, model, compress, timeline, qa/, eval/)
scripts/            download, build_dataset, train, tune_smoothing, evaluate, benchmark, answer, make_synthetic
tests/              pytest checks for every layer
configs/split.json  fixed train/test users
report/             report outline and figures
models/, results/   outputs (models are not committed)
```

## Roadmap

- [x] End-to-end baseline: preprocessing → HGB → timeline → rule parser → formatted grounded answers
- [x] Evaluation harness: question generator, scoring rules, Figures 1, 2, 3, 5
- [x] Run on real ExtraSensory data and do error analysis on the confusion matrix
- [x] Sticky-HMM temporal decoding against timeline fragmentation (tuned on training users)
- [ ] Compact 1D-CNN backbone (compare with HGB)
- [ ] Optional SLM front end (Qwen2.5-0.5B/1.5B or Llama-3.2-1B) that emits the same Intent JSON, rule parser as fallback
- [x] Efficiency: full vs quantised/pruned/distilled models, `benchmark.py`, Figure 4 (Pareto front)
- [ ] Hand-written hard and edge-case question set; open-world rubric grading (1–5)
- [ ] Report, contribution statement, AI-use disclosure

## Citations

- Dataset: Y. Vaizman, K. Ellis, G. Lanckriet. *Recognizing Detailed Human Context In-the-Wild from
  Smartphones and Smartwatches.* IEEE Pervasive Computing 16(4), 2017. http://extrasensory.ucsd.edu
- Libraries: NumPy, pandas, SciPy, scikit-learn, matplotlib, joblib (versions in `requirements.txt`).

## AI-use disclosure

Claude Code (Anthropic) was used to scaffold the initial repository structure and baseline code.
Each group member must review, understand and own the code in their area; record further AI use here.
