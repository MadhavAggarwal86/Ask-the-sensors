# Report outline (target 10–12 pages)

1. **Problem framing** (1 p): the sensor QA task, why grounding matters, the four tiers.
2. **Data** (1 p): ExtraSensory subset, 7 classes, class imbalance (labelled minutes: sitting 136k,
   lying 104k, standing+moving 30k, walking 23k, standing in place 8k, bicycling 5k, running 1.3k),
   20 s per minute recording, missing gyro, self-reported labels. Our split (leave users out).
3. **Design** (2–3 p): the four layers and why.
   - Why engineered features + HGB first (small, explainable, NaN-tolerant, edge-friendly)
   - Why the timeline is the single source of truth (no numbers from language models)
   - Window 6 s / hop 3 s, smoothing, gap bridging (45 s), bout and "prolonged" definitions
   - Question parser (rules, then optional SLM producing the same Intent JSON)
   - Explanations as templates over measured features, which keeps them faithful
   - Open-world reasoning: behaviours described by supporting activities + signal properties
4. **Implementation** (1 p): repo structure, how to reproduce, tests.
5. **Results** (3 p)
   - Fig 1 accuracy by question type (with correctness rule per type)
   - Fig 2 confusion matrix + per-class P/R/F1
   - Fig 3 accuracy vs strictness
   - Fig 5 robustness curve
   - Answer accuracy vs grounded accuracy (the "price of evidence")
   - Error analysis: typical failure examples
6. **Efficiency** (1–2 p, extra credit): target hardware, size/memory/latency, Fig 4 Pareto front.
7. **Limitations and future work** (0.5 p)
8. **Contribution statement** (per member)
9. **AI-use disclosure**
