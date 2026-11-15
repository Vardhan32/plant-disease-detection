# Experiment configurations

Each JSON file is a complete `backend/train.py` configuration. Config values
not present in a file fall back to the defaults in `backend/train.py`.

All experiments:

- use the committed split manifest `backend/data/plantvillage_split.csv`
  (14,444 train / 3,097 validation / 3,097 test images),
- use seed 42,
- select the model on validation data only,
- set `"promote": false`, so the active serving model is not replaced until a
  run is explicitly promoted with `backend/promote_model.py`.

| Config | Resolution | Augmentation | Class weights | Purpose |
|---|---|---|---|---|
| `baseline_128_frozen.json` | 128 | none | none | Reproduce the original recipe on the new split |
| `baseline_224_frozen.json` | 224 | none | none | Measure the effect of resolution |
| `aug_224_frozen.json` | 224 | moderate | none | Measure the effect of augmentation |
| `aug_224_frozen_cw.json` | 224 | moderate | square-root inverse frequency | Measure the effect of class weighting |
| `finetune_224_aug.json` | 224 | moderate | none | Fine-tune the top base-model layers from the best frozen run |

Run one experiment from the repository root:

```bash
python backend/train.py --config backend/configs/baseline_128_frozen.json
```

The test split is never read during training. After selecting a model on
validation metrics, evaluate the chosen run exactly once:

```bash
python backend/evaluate.py --split test --allow-test \
  --model backend/runs/<run-name>/best_model.h5 \
  --output-dir backend/reports/<run-name>_test
```

Then promote it:

```bash
python backend/promote_model.py --run-dir backend/runs/<run-name>
```
