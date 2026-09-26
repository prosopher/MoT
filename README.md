# KV Cache Translation across Heterogeneous Large Language Models

## Overview

This repository studies KV-cache translation across heterogeneous Large Language Models (LLMs), enabling a context cache produced by one model to be reused by another model with a different cache space. Our method, **MoT (Mixture-of-Translators)**, uses token-level routing over multiple translators together with a **Context Correction Loss** to improve translation quality while supporting memory-efficient cache reuse in heterogeneous multi-model workflows.

## Installation
```console
pip install -r requirements.txt
```

## Train
```console
python train.py mot
python train.py mot --default-config-path configs/train_mot_toy.json
python train.py mot --default-config-path configs/train_mot_toy.json --top-layers-to-translate 3
python train.py lsc
python train.py lsc --default-config-path configs/train_lsc_toy.json
```

## Evaluation
```console
python eval.py mot
python eval.py mot --default-config-path configs/eval_toy.json
```

## Case Study - Multi Agent Reasoning (StrategyQA Memory)
```cmd
python exp/multi_agents_qa.py <algorithm> --checkpoint-dir-path <checkpoint_dir> --cache-mode <retain|free> --agent-count <N> --max-turns <T> --max-examples <M>

python exp/multi_agents_qa.py mot --checkpoint-dir-path outputs/mot_... --cache-mode retain --agent-count 4 --max-turns 7 --max-examples 30
```
