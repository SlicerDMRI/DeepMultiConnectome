# Analysis

Run the scripts below to reproduce the paper figures and summary tables.

## Scripts

1) Intra/inter-subject similarity

```bash
python analysis/compute_connectome_similarities.py --subject-list /path/to/subjects_test.txt
```

Outputs: per-subject metrics CSV/JSON and summary plots under `analysis/`.

2) Population-average comparison

```bash
python analysis/create_population_connectomes.py --subject-list /path/to/subjects_train.txt
```

Outputs: population averages and test-vs-population summaries under `analysis/`.

3) Test-retest similarity

```bash
python analysis/compute_trt_similarity.py --subjects_file /path/to/subjects_trt.txt
```

Outputs: TRT similarity tables and plots under `analysis/`.

4) Network metrics similarity (pred vs true)

```bash
python analysis/compute_network_metric_similarity.py --subject-list /path/to/subjects_test.txt
```

Outputs: per-subject and summary CSVs under `analysis/`.

## Batch runner

```bash
bash analysis/run_all_analyses.sh
```

## Shared metrics

Common metrics live in `analysis/utils/analysis_metrics.py`.
