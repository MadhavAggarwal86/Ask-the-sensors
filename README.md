# Ask the Sensors

Grounded, explainable activity question answering from wearable accelerometer + gyroscope signals
(ExtraSensory, 7 classes: lying down, sitting, standing in place, standing and moving, walking,
running, bicycling).

**Time base:** every timestamp is reported in **seconds from the start of the recording**.

## Results at a glance (12 held-out users, real ExtraSensory data)

| | |
|---|---|
| Overall QA accuracy (macro over 8 question types, 767 questions, 40 recordings) | **0.636** (0.553 before HMM decoding) |
| Answer correct **and** evidence grounded | **0.475** (0.381 before) |
| Parser intent accuracy on unseen hand-written questions | **0.96** (rules) vs 0.58 (Qwen2.5-0.5B) |
| Verification (yes/no) precision / recall / specificity | 0.88 / 0.60 / 0.92 |
| Duration error, median (mean) | 163 s (1,056 s); count error median 1 bout |
| Distilled int8 model vs full model | **380× smaller** (19 KB vs 7.3 MB, 13.8 k vs 128 k parameters), accuracy difference −0.020 to +0.017 (95% CI) |
| Cost per query, 1-hour recording, 1 M2 core | 94 ms and ~367 mJ (full) → 46 ms and ~178 mJ (int8); energy estimated as CPU time × 4 W |

Full write-up: [`report/report.md`](report/report.md) (build PDF/Word with `report/build_report.sh`).

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
| Answer | `src/ats/qa/` | `parser.py` question → intent (rules); `llm.py` optional small-LLM / hybrid parser; `operations.py` intent → answer + evidence; `explain.py` explanations built only from measured values; `format.py` output format |
| Evaluate | `src/ats/eval/`, `scripts/evaluate.py`, `scripts/eval_parser.py` | auto-generated questions from ground-truth labels, scoring rules per type, Figures 1, 3, 5; intent accuracy on hand-written question sets (`eval_sets/`) |
| Compress | `src/ats/compress.py`, `scripts/benchmark.py` | boosting-round pruning, distillation into a tree / small MLP, int8 quantisation; accuracy vs size and latency, Figure 4 |

**Grounding rule:** the language layer never produces a number or timestamp. Every value comes from
the timeline, and every explanation is a template filled with feature values of the cited windows.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install -r requirements-llm.txt   # optional: small-LLM parser (model downloaded on first use)
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

# 6. Efficiency: pruned / distilled / int8 / accelerometer-only variants, Figure 4 (Pareto front);
#    parameters, CPU time and an energy estimate are included (--costs-only re-measures just those)
python scripts/benchmark.py --source extrasensory

# 7. Parser generalisation on the hand-written question sets (add --parser llm for the SLM)
python scripts/eval_parser.py --split both

# 8. Explanation rubric (1-5): one grading sheet per person, then agreement between graders
python scripts/grade_explanations.py sheet --grader grader_A
python scripts/grade_explanations.py agree results/extrasensory/rubric/rubric_grader_*.csv
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
question per line and `--json out.jsonl` for machine-readable output. `--parser hybrid` lets a small
LLM take questions the rules cannot route (needs the `llm` extra). `eval_sets/demo_questions.txt`
is a demo set covering all four tiers.

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
scripts/            download, build_dataset, train, tune_smoothing, evaluate, benchmark,
                    eval_parser, grade_explanations, answer, make_synthetic
tests/              pytest checks for every layer
configs/split.json  fixed train/test users
eval_sets/          hand-written parser dev/test sets (annotated intents) and demo questions
report/             report source (report.md), architecture figure, build script
models/, results/   outputs (models are not committed)
```

## Roadmap

- [x] End-to-end baseline: preprocessing → HGB → timeline → rule parser → formatted grounded answers
- [x] Evaluation harness: question generator, scoring rules, Figures 1, 2, 3, 5
- [x] Run on real ExtraSensory data and do error analysis on the confusion matrix
- [x] Sticky-HMM temporal decoding against timeline fragmentation (tuned on training users)
- [x] Efficiency: full vs quantised/pruned/distilled models, `benchmark.py`, Figure 4 (Pareto front)
- [x] Hand-written question sets (dev/test) and parser hardening; unmeasurable-quantity refusals; fall signature
- [x] Optional SLM front end (Qwen2.5-0.5B; 1.5B does not fit in 8 GB RAM) emitting the same Intent JSON, validated, rules as fallback
- [x] Report draft with all five required figures
- [x] Answer-level metrics (P/R/F1/specificity, macro-F1, MAE), parameter counts, CPU time, energy estimate
- [x] Explanation rubric and grading sheets (a 1.5B LLM judge was piloted and rejected, see report 5.7)
- [ ] Group: grade the two rubric sheets in `results/extrasensory/rubric/` and add the scores to report 5.7
- [ ] Group: fill in names, contribution statement and AI-use disclosure in the report
- [ ] Future work: noise-augmented training; compact 1D-CNN backbone; onset change-point refinement

## Citations

- Dataset: Y. Vaizman, K. Ellis, G. Lanckriet. *Recognizing Detailed Human Context In-the-Wild from
  Smartphones and Smartwatches.* IEEE Pervasive Computing 16(4), 2017. http://extrasensory.ucsd.edu
- Libraries: NumPy, pandas, SciPy, scikit-learn, matplotlib, joblib (versions in `requirements.txt`);
  optional PyTorch and Hugging Face Transformers with Qwen2.5-Instruct (Qwen Team, 2024) for the LLM parser.

## AI-use disclosure

Claude Code (Anthropic) was used to draft and refactor code across all layers, diagnose data
problems, design and run the experiments, generate figures and draft the report. Each group member
must review, understand and own the code in their area; the full statement is in the report
(Section 9) and must be completed by the group before submission.
