# KVComm vendor snapshot

This repository keeps the original KVComm model-side implementation snapshot under `kvcomm/vender/`, but the integrated evaluation path does **not** use the old vendor `dataloader` package anymore.

## Supported path in this repo

Use the repository-level CLI so KVComm is evaluated the same way as the other algorithms:

```bash
python eval.py kvcomm --default-config-path configs/eval.json
```

KVComm checkpoints are produced through the repository-level training entrypoint:

```bash
python train.py kvcomm --default-config-path configs/train_kvcomm.json
```

## Removed legacy path

The standalone vendor evaluation entrypoints that depended on `kvcomm/vender/dataloader` were intentionally removed.
This repository does not provide backward compatibility for that flow.
