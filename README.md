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

## Longcontext 시각화
사용 파일:
- `longcontext_ttft_128/model_budget_table_with_upperbound.csv`: budget 기준 원본 요약 데이터
- `plot_eval_results.py`: line/bar/table 전체 집계 시각화
- `longcontext_budget_eval_plot.py`: 논문용 avg/budget 플롯 생성

전체 집계 시각화:
```cmd
python plot_eval_results.py --budget-csv longcontext_ttft_128/model_budget_table_with_upperbound.csv --output-dir outputs_eval/ttft_128 --modes line avg_bar avg_table budget_table --models upperbound c2c interlat mot128
```

논문용 플롯(내장 데이터 사용):
```cmd
python longcontext_budget_eval_plot.py --input-source embedded --plot-types avg budget --models upperbound c2c interlat mot128 --budgets 4096 8192 16384 24576 --output-dir longcontext_ttft_128/paper_plots_overleaf
```

논문용 플롯(CSV 입력 사용):
```cmd
python longcontext_budget_eval_plot.py --input-source csv --budget-csv longcontext_ttft_128/model_budget_table_with_upperbound.csv --plot-types avg budget --models upperbound c2c interlat mot128 --max-budgets 3 --output-dir longcontext_ttft_128/paper_plots_overleaf
```

주요 출력 파일:
- `avg_f1_ttft.(png|pdf)`, `budget_f1.(png|pdf)`, `budget_ttft.(png|pdf)`
- `eval_plot_2paper_average_table.(csv|md)`
