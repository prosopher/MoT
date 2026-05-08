# Mixture-of-Translators

## 설치
```console
pip install -r requirements.txt
```

## 학습
```console
python train.py mot
python train.py mot --default-config-path configs/train_mot_toy.json
python train.py mot --default-config-path configs/train_mot_toy.json --top-layers-to-translate 3
python train.py lsc
python train.py lsc --default-config-path configs/train_lsc_toy.json
```

## 평가
```console
python eval.py
python eval.py --default-config-path configs/eval_toy.json
```

## Longcontext 샘플 명령어
```cmd
python plot_eval_results.py --budget-csv longcontext_ttft_128/model_budget_table_with_upperbound.csv --output-dir outputs_eval/ttft_128 --modes line avg_bar avg_table budget_table --models upperbound c2c interlat mot128

python longcontext_budget_eval_plot.py --input-source embedded --plot-types avg budget --models upperbound c2c interlat mot128 --budgets 4096 8192 16384 24576 --output-dir longcontext_ttft_128/paper_plots_overleaf

python longcontext_budget_eval_plot.py --input-source csv --budget-csv longcontext_ttft_128/model_budget_table_with_upperbound.csv --plot-types avg budget --models upperbound c2c interlat mot128 --max-budgets 3 --output-dir longcontext_ttft_128/paper_plots_overleaf
```
