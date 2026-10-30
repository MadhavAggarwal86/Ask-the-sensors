# Ask the Sensors

Grounded, explainable activity question answering from wearable accelerometer + gyroscope signals

(ExtraSensory, 7 classes: lying down, sitting, standing in place, standing and moving, walking,

running, bicycling).

**Time base:** every timestamp is reported in **seconds from the start of the recording**.

## Quick start (web page, about 5 minutes)

The trained activity model and a demo recording are included, so no dataset download is needed:

```bash
git clone https://github.com/MadhavAggarwal86/Ask-the-sensors.git && cd Ask-the-sensorsAsk-the-sensors
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
CMAKE_ARGS="-DGGML_METAL=on" pip install -r requirements-llm.txt
python -c "from huggingface_hub import hf_hub_download as d; d('unsloth/Qwen3-1.7B-GGUF', 'Qwen3-1.7B-Q4_K_M.gguf', local_dir='models/slm')"
streamlit run app.py
```

## Results at a glance (12 held-out users, real ExtraSensory data)

| | |
|---|---|
| Overall QA accuracy (macro over 8 question types, 767 questions, 40 recordings) | **0.612** with the Qwen3-1.7B parser |
| Answer correct **and** evidence grounded | **0.462** |
| Question understanding (Qwen3-1.7B, 4-bit, llama.cpp; intent accuracy on hand-written questions) | **0.84** dev / **0.82** test, ~2 s per question |
| Verification (yes/no) precision / recall / specificity | 0.85 / 0.73 / 0.87 |
| Duration error, median (mean) | 273 s (1,107 s); count error median 1 bout |
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
| Answer | `src/ats/qa/` | `llm.py` the language model (Qwen3-1.7B, 4-bit, llama.cpp) turns the question into an intent, extracts time expressions and words the explanation; `operations.py` intent → answer + evidence; `explain.py` explanations built only from measured values; `format.py` output format |
| Evaluate | `src/ats/eval/`, `scripts/evaluate.py`, `scripts/eval_parser.py` | auto-generated questions from ground-truth labels, scoring rules per type, Figures 1, 3, 5; Qwen3 intent accuracy on hand-written question sets (`eval_sets/`) |
| Compress | `src/ats/compress.py`, `scripts/benchmark.py` | boosting-round pruning, distillation into a tree / small MLP, int8 quantisation; accuracy vs size and latency, Figure 4 |
**Grounding rule:** the language layer never produces a number or timestamp. Every value comes from

the timeline, and every explanation is a template filled with feature values of the cited windows.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
CMAKE_ARGS="-DGGML_METAL=on" pip install -r requirements-llm.txt   # llama.cpp + Streamlit
# the language model, Qwen3-1.7B quantised to 4 bits (1.0 GB):
python -c "from huggingface_hub import hf_hub_download as d; d('unsloth/Qwen3-1.7B-GGUF', 'Qwen3-1.7B-Q4_K_M.gguf', local_dir='models/slm')"
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
#    temporal decoder can be configured through the evaluation options
python scripts/evaluate.py --source extrasensory --robustness
# 6. Efficiency: pruned / distilled / int8 / accelerometer-only variants, Figure 4 (Pareto front);
#    parameters, CPU time and an energy estimate are included (--costs-only re-measures just those)
python scripts/benchmark.py --source extrasensory
# 7. Evaluate Qwen3 question understanding on the hand-written question sets
python scripts/eval_parser.py --split both --parser llm
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

question per line and `--json out.jsonl` for machine-readable output, `--explainer template` for the

fixed-sentence explanation, and `--slm Qwen3-0.6B` for the smaller, faster language model (default

`Qwen3-1.7B`; download its file `Qwen3-0.6B-Q4_K_M.gguf` the same way as in Setup). `eval_sets/demo_questions.txt` is a demo set covering all four tiers.

How a question is answered: the language model reads it and fills a fixed JSON schema (operation,

activities, the time expression as written, open-world topic); llama.cpp enforces the schema, and a

time or day period that is not in the question is dropped. Python converts the time to seconds and

computes the answer and evidence from the sensor timeline. The model then words the explanation;

the wording is kept only if every number in it was computed by Python.

### Web page

```bash
streamlit run app.py
```

Choose the language model in the sidebar (Qwen3-1.7B, more accurate; Qwen3-0.6B, faster), upload

one or more recording CSVs (or use the demo recording), pick a recording, type a question:

the page shows the six-field answer (Answer, Activity/Event, Timestamps, Sensor Modality, Sensor

Channels, Explanation), the JSON form and the recognised activity timeline. The recording is

processed once; each question then takes about 2–5 s on an M2 laptop.

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
app.py              Streamlit web page (upload a recording, ask, get the six-field answer)
tests/              pytest checks for every layer
configs/split.json  fixed train/test users
eval_sets/          hand-written parser dev/test sets (annotated intents) and demo questions
report/             report source (report.md), architecture figure, build script
models/, results/   outputs (models are not committed)
```

## Roadmap

- [x] End-to-end pipeline: preprocessing → HGB → timeline → language-model parser → formatted grounded answers

- [x] Evaluation harness: question generator, scoring rules, Figures 1, 2, 3, 5

- [x] Run on real ExtraSensory data and do error analysis on the confusion matrix

- [x] Sticky-HMM temporal decoding against timeline fragmentation (tuned on training users)

- [x] Efficiency: full vs quantised/pruned/distilled models, `benchmark.py`, Figure 4 (Pareto front)

- [x] Hand-written question sets (dev/test), unmeasurable-quantity refusals and fall signature

- [x] Language-model front end: Qwen3-1.7B (4-bit GGUF, llama.cpp) extracts the intent with a schema-constrained grammar and words the explanation; Python computes every number

- [x] Streamlit web page

- [x] Report draft with all five required figures

- [x] Answer-level metrics (P/R/F1/specificity, macro-F1, MAE), parameter counts, CPU time, energy estimate

- [x] Explanation rubric and grading sheets (a 1.5B LLM judge was piloted and rejected, see report 5.7)

- [ ] Group: grade the two rubric sheets in `results/extrasensory/rubric/` and add the scores to report 5.7

- [ ] Future work: noise-augmented training; compact 1D-CNN backbone; onset change-point refinement

## Citations

- Dataset: Y. Vaizman, K. Ellis, G. Lanckriet. *Recognizing Detailed Human Context In-the-Wild from

  Smartphones and Smartwatches.* IEEE Pervasive Computing 16(4), 2017. http\://extrasensory.ucsd.edu

- Libraries: NumPy, pandas, SciPy, scikit-learn, matplotlib, joblib (versions in `requirements.txt`);

  llama.cpp (llama-cpp-python) with Qwen3-1.7B (Qwen Team, 2025; 4-bit GGUF by Unsloth) for question

  understanding and explanations, and Streamlit for the web page.
