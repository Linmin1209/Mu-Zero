# RoboCasa365 Task Semantic Analysis

## Corpus stats
- Unique tasks: 317
- Total description variants: 6270
- Mean variants per task: 18.5 (median 2)
- 1-word descriptions (task name only): 352 / 6270 (5.6%)

## Inter-task similarity (TF-IDF, primary text)
- All tasks: mean=0.0452, median=0.0290, p90=0.1000, max=0.8242
- Eval 50:   mean=0.0402, median=0.0227, p90=0.0895, max=0.7188

## Gate vs semantics (eval)
- corr(gate_delta, max_sim_eval) = 0.174
- corr(motion, primary_word_count) = -0.530

## Figures
- `01_similarity_histogram.png` — global pairwise similarity distribution
- `02_family_similarity_boxplot.png` — similarity by task family
- `03_word_count_distribution.png` — instruction length
- `04_eval_heatmap.png` — 50 eval tasks similarity matrix
- `05_eval_clustermap.png` — hierarchical clustering
- `06_tsne_all_tasks.png` — 2D embedding (eval tasks highlighted)
- `07_gate_vs_semantics.png` — gate delta vs language features
- `08_top_similar_pairs.png` — highest-similarity pairs