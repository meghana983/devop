---
name: testing-visualizations
description: Test generated model evaluation visualizations in results_fixed. Use when verifying changes to generate_visualizations.py or generated ROC/confusion matrix/fold plots.
---

# Testing generated visualizations

## Devin Secrets Needed

None. This workflow runs locally and does not require login, API keys, or external services.

## Setup

From the repo root, install plotting dependencies:

```bash
python3 -m pip install --user -r requirements.txt
```

## Primary test flow

1. Regenerate plots:

```bash
python3 generate_visualizations.py
```

Expected output includes `Saved fold_1 visualizations` through `Saved fold_5 visualizations`, followed by `All visualizations saved to .../results_fixed/`.

2. Verify generated artifact structure:

```bash
python3 - <<'PY'
from pathlib import Path
from PIL import Image

root = Path('results_fixed')
fold_expected = [
    'confusion_matrix.png',
    'confusion_matrix_normalized.png',
    'metrics_visualization.png',
    'roc_curve.png',
]
pngs = sorted(root.glob('**/*.png'))
assert len(pngs) == 27, len(pngs)
for fold_idx in range(1, 6):
    fold_dir = root / f'fold_{fold_idx}'
    assert sorted(p.name for p in fold_dir.glob('*.png')) == fold_expected
    for name in fold_expected:
        with Image.open(fold_dir / name) as img:
            assert img.format == 'PNG'
            assert img.size[0] > 0 and img.size[1] > 0
PY
```

3. Capture visual evidence for a test report. Include representative images such as `results_fixed/fold_1/metrics_visualization.png`, `results_fixed/fold_1/roc_curve.png`, `results_fixed/fold_1/confusion_matrix.png`, and `results_fixed/summary_overview.png`.

## Notes

- This is shell-only testing; do not record the desktop unless a future UI is added.
- If expected fold counts change, update the exact assertions before testing rather than weakening them.
