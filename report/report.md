\---

title: "Ask the Sensors: Grounded, Explainable Activity Question Answering over Wearable Motion Signals"

author: "Jyotirmoy Konwar, Venu Gopal Batulla, Madhav Aggarwal, Aryan Khaiwal"

date: "30th October 2026"


\---
*** A deterministic parser built first (spoken-form normalisation, typo

correction, time patterns, ordered operation patterns; 0.3 ms per question) is kept as a baseline

and used where an experiment should isolate the sensor pipeline (Sections 5.3, 5.4 and 6).

**## 3.6 Operations and the grounding rule**

Each operation reads the timeline and fills the six required output fields:

\| Operation | Answer | Evidence cited |

\|---|---|---|

\| identify | activity with the most time in the range | its longest intervals in the range |

\| verify | Yes / Likely yes / No | the matching bouts; for "No", the dominant activity instead |

\| duration | total seconds | every contributing interval |

\| count | number of bouts ≥ 10 s | the bouts |

\| onset | "Yes, X began at t s" | the first bout, contrasted with the preceding interval |

\| compare | the activity with more (or less) time | both totals |

\| longest | length and span of the longest bout | that bout |

\| prolonged | Yes/No against the 600 s threshold | the longest bout |

\| open | verdict (Consistent / Partially / Not consistent) | intervals of supporting activities |

: Operations. All numbers are read off the timeline.

The modality field is "Accelerometer, Gyroscope" when the recording has a gyroscope and

"Accelerometer" otherwise. Explanations are templates filled with the mean feature values of the

cited windows, contrasted with the rest of the recording, for example: \*"The cited windows show

accelerometer magnitude variability of 0.193 g, a regular rhythm at 2.14 Hz (about 128 cycles/min,

periodicity 0.39), gyroscope rotation averaging 0.97 rad/s … For contrast, the rest of the recording

averages 0.062 g variability."\* A "Yes" whose supporting intervals have a mean probability below 0.6

is hedged as "Likely yes", and a "No" is still grounded by citing what the user was doing instead.

The language model then rewords this template in plain sentences from the computed fields only;

its text is kept only if every number in it appears in those fields (otherwise the template is used:

14 of 670 explanations), and "N/A" answers are never reworded.

**## 3.7 Open-world reasoning**

Each behaviour outside the seven labels is described by the recognised activities that would

support it and a signal property it requires: exercise needs running, cycling or walking at

moderate-to-vigorous intensity; commuting needs sustained walking or cycling bouts (≥ 5 min);

sleeping needs lying down with almost no motion; desk work needs still sitting; chores and shopping

need light standing-and-moving and walking. The verdict follows from the share of analysed time

that meets both conditions (≥ 50% consistent, ≥ 15% partially), and the explanation states that

share, the intensity and the longest bout. Falls are handled separately: a fall needs an impact

(peak > 2 g with jerk far above the recording's typical level) **\*\*followed by\*\*** a fall signature —

stillness (< 0.03 g variability) or a device-orientation change above 45° within the next minute.

Impacts during continuing movement with unchanged orientation are reported as likely footstrikes

or bumps, with their timestamps, rather than as falls.

**# 4. Implementation and reproducibility**

The code is a Python package (\`src/ats/\`) with one module per layer, plus scripts:

\| Step | Command | Output |

\|---|---|---|

\| Download raw data (\~16 GB zipped) | \`scripts/download_data.py\` | \`data/raw/\` (never committed) |

\| Window features per user | \`scripts/build_dataset.py\` | \`data/processed/\` |

\| Train, Figure 2 | \`scripts/train.py\` | \`models/hgb_full.joblib\` |

\| Tune the decoder (optional) | \`scripts/tune_smoothing.py\` | tuning table and figure |

\| QA evaluation, Figures 1, 3, 5 | \`scripts/evaluate.py --robustness\` | metrics, records, figures |

\| Efficiency, Figure 4 | \`scripts/benchmark.py\` | \`efficiency.json\`, figure |

\| Parser evaluation | \`scripts/eval_parser.py\` | intent accuracy per set |

\| Explanation rubric | \`scripts/grade_explanations.py sheet\` / \`agree\` | grading sheets, agreement |

\| Answer questions | \`scripts/answer.py --recording r.csv --questions q.txt [--slm Qwen3-0.6B]\` | 6-field answers |

\| Web page | \`streamlit run app.py\` | upload, choose model, ask |

All randomness is seeded: the user split is stored in \`configs/split.json\`, question sampling is

seeded by a CRC32 hash of the recording id (Python's \`hash()\` is salted per process), and models use

fixed random states. A test suite of 25 checks (\`pytest\`) exercises every layer on synthetic

signals, which are used only for testing and never reported as results. The runnable system reads

any CSV with a time column and accelerometer (optionally gyroscope) columns, with loose column-name

matching; a 4.4-hour recording is processed in under 1 s, after which each question takes 2–5 s,

almost all of it the language model. A Streamlit page (\`app.py\`) offers the same pipeline: upload one

or more recordings, choose Qwen3-1.7B or Qwen3-0.6B, ask a question and read the six-field answer

with the recognised activity timeline.

**# 5. Results**

**## 5.1 Recognition backbone**

![Figure 2. Window-level confusion matrix on the 12 test users (raw classifier output, before temporal decoding).]\(../results/extrasensory/fig2_confusion_hgb_full.png){width=44%}

\| Class | Precision | Recall | F1 |

\|---|---:|---:|---:|

\| Lying down | 0.49 | 0.33 | 0.40 |

\| Sitting | 0.60 | 0.58 | 0.59 |

\| Standing in place | 0.03 | 0.06 | 0.04 |

\| Standing and moving | 0.10 | 0.16 | 0.13 |

\| Walking | 0.47 | 0.57 | 0.52 |

\| Running | 0.01 | 0.01 | 0.01 |

\| Bicycling | 0.34 | 0.60 | 0.43 |

: Per-class window metrics on test users. Macro-F1 0.30 raw, 0.38 after HMM decoding; balanced accuracy 0.33 raw, 0.36 decoded.

Window-level accuracy is modest, and the confusion matrix shows why. (i) **\*\*Lying versus sitting\*\***:

both are still, so they are told apart only by phone orientation, which depends on where the phone

is (pocket, bag, table); 43% of lying windows are predicted as sitting. (ii) **\*\*Label noise\*\***: 816 of

the 1,087 test windows labelled running come from one user whose phone was essentially motionless

(median variability 0.002 g) during those minutes, so no motion-based model can recognise them.

Restricted to windows where the phone actually moves (variability ≥ 0.05 g), walking recall is

0.81 and bicycling 0.83. (iii) **\*\*Running versus walking\*\***: even on moving windows only 6% of

running is recognised; 76% is called walking, as a phone in a bag does not show the larger impacts

a body-worn sensor would. Vaizman et al. score each label one-vs-rest and fuse more sensors

(watch, audio, location), so their figures are not directly comparable with a 7-way choice from

phone motion alone. For reference, our one-vs-rest balanced accuracies are 0.79 (bicycling), 0.75

(walking), 0.59–0.60 (sitting, lying) and 0.50–0.52 (standing in place, standing and moving,

running).

**## 5.2 Question answering**

![Figure 1. QA accuracy of the full system (Qwen3-1.7B parser) by question type on 767 questions over 40 held-out recordings, with the correctness rule under each type. Blue: answer correct; orange: answer correct and evidence grounded.]\(../results/extrasensory/fig1_accuracy_by_type_hgb_full_hmm_slm_slm.png){width=68%}

\| Answer kind | Metric | SLM system |
|---|---|---|
| Verification (yes/no, n = 150) | P / R / F1 on "yes"; specificity | **0.85 / 0.73 / 0.79; 0.87** |
| Prolonged (yes/no, n = 80) | P / R / F1 on "yes"; specificity | **0.70 / 0.92 / 0.79; 0.82** |
| Onset detected (n = 74) | recall | **0.95** |
| Identification (4 classes, n = 75) | macro-F1 / balanced accuracy | **0.46 / 0.44** |
| Comparison (6 winners, n = 80) | macro-F1 / balanced accuracy | **0.70 / 0.74** |
| Open-world (3 verdicts, n = 80) | macro-F1 / balanced accuracy | **0.67 / 0.69** |
| Duration (n = 114) | mean / median abs. error; MAPE | **1,107 s / 273 s; 255%** |
| Count (n = 114) | mean / median abs. error | **4.9 / 1.0 bouts** |
| Cited evidence (duration, onset) | grounding precision | **0.40** |

: Answer-level metrics for the SLM system on the same 767 questions.

These metrics expose biases that accuracy hides. Verification and prolonged questions remain affected by under-recognised postures and over-merged stationary intervals. Numeric errors are heavy-tailed: posture misclassification in long recordings can substantially increase duration error. With the language model the median duration error rises to 273 s and MAPE to

255%, because a misread time range makes the system measure the wrong stretch of the recording;

MAPE is in any case inflated by references of a few minutes.

**## 5.3 Accuracy versus strictness**

![Figure 3. Left: share of duration answers accepted as the relative-error tolerance grows. Right: share of cited intervals accepted as the IoU threshold grows.]\(../results/extrasensory/fig3_strictness_hgb_full_hmm.png){width=58%}

Sections 5.3 and 5.4 measure the sensor pipeline under a fixed question interpretation. Figure 3

shows whether misses are near or wild. Duration acceptance grows steadily with the

tolerance, so many misses are moderate rather than catastrophic. For onsets, the yes/no part is

right in 82% of cases; when both system and reference say yes, the predicted onset is within 60 s

of the true one in 34% of cases, within 5 minutes in 51% (median error 258 s). The fixed IoU ≥ 0.5

rule is therefore strict for onsets: the first bout is often found a few minutes early or late, or

starts at the very beginning of a recording that the reference begins later.

**## 5.4 Robustness**

![Figure 5. Overall QA accuracy on a fixed set of 10 recordings as the input is degraded.]\(../results/extrasensory/fig5_robustness_hgb_full_hmm.png){width=62%}

\| Degradation | Levels → overall QA accuracy |

\|---|---|

\| Dropped samples | 0%: 0.67 · 10%: 0.66 · 30%: 0.65 · 50%: 0.65 · 70%: 0.63 |

\| Sampling rate | 25 Hz: 0.67 · 15 Hz: 0.65 · 10 Hz: 0.64 · 5 Hz: 0.61 |

\| Gaussian noise | 0 g: 0.67 · 0.02 g: 0.62 · 0.05 g: 0.61 · 0.1 g: 0.29 · 0.2 g: 0.28 |

: Robustness (HMM decoder). Recordings are re-resampled from the surviving samples, as the loader would treat a sparser stream.

The system degrades gracefully with missing data and low sampling rates: 70% dropped samples cost

0.04, and sampling at 5 Hz costs 0.06. Gap-aware resampling and the per-window coverage rule make

missing data a non-issue, and the features that matter (variability, tilt, cadence up to \~2.5 Hz)

survive at 5 Hz. Additive noise is the weak point: up to 0.05 g accuracy holds (the HMM decoder

raised accuracy at 0.05 g from 0.26 to 0.61), but at 0.1 g and above accuracy collapses, because

still postures are separated by variability differences of a few milli-g, which the noise swamps.

Typical phone noise is a few milli-g, so the collapse lies outside realistic conditions.

**## 5.5 Parser generalisation**

**# 8. Contribution statement**

\| Member | Contributions |

\|---|---|

\| Madhav Aggarwal | **\*\*Data and preprocessing.\*\*** ExtraSensory loader, unit and timestamp cleaning, faulty-sensor exclusion, 25 Hz resampling with gap masking, 6 s windows, user split. |

\| Jyotirmoy Konwar | **\*\*Language model.\*\*** Qwen3 via llama.cpp, intent schema and prompt, time conversion and grounding checks, explanation rewording with number check, question understanding and evaluation. |

\| Venu Gopal Battula | **\*\*Evaluation.\*\*** Generated QA set and scoring, hand-written question sets, answer-level metrics, evaluation runs and Figures 1, 3, 5, error analysis, explanation rubric. |

\| Aryan Khaiwal | **\*\*Deployment.\*\*** Streamlit page (\`app.py\`) with upload, model choice, six-field answer and timeline; command-line script. |


**# References**

\- Y. Vaizman, K. Ellis, G. Lanckriet. Recognizing detailed human context in the wild from

&#x20; smartphones and smartwatches. *\*IEEE Pervasive Computing\** 16(4), 2017.

\- J. Ba, R. Caruana. Do deep nets really need to be deep? *\*NeurIPS\**, 2014.

\- G. Ke et al. LightGBM: a highly efficient gradient boosting decision tree. *\*NeurIPS\**, 2017

&#x20; (the histogram-based boosting that scikit-learn's HistGradientBoosting follows).

\- L. Rabiner. A tutorial on hidden Markov models and selected applications in speech recognition.

*\*Proceedings of the IEEE\** 77(2), 1989.

\- Qwen Team. Qwen2.5 technical report, arXiv:2412.15115, 2024; Qwen3 technical report, arXiv:2505.09388, 2025.

\- G. Gerganov et al. llama.cpp, https\://github.com/ggml-org/llama.cpp (GGUF quantisation of Qwen3 by Unsloth).
