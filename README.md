# Process Unlearning Dashboard

Interactive Streamlit dashboard to choose a process (P0–P9) to **delete/unlearn** from the LSTM model and view metrics.

## Quick start

### 1. Create a virtual environment (optional)

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

A browser window will open with the dashboard. Select a process, click **Unlearn process**, and view the results.

---

## What it does

- **Select process**: Choose P0–P9 in the sidebar
- **Unlearn**: Trains the model, then runs unlearning (erase target + restore others)
- **Metrics**:
  - **Certified Forgetting Score**: RMSE increase on the target (higher = more forgetting)
  - **Baseline / After / Comparison** tables (RMSE, MAE, R²)

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard |
| `core.py` | Unlearning pipeline (parameterized by `target_process`) |
| `requirements.txt` | Python dependencies |

---

## Notebook changes

To make `fullcodetry2 (1).ipynb` use a **selectable** target process instead of hardcoded `"P3"`:

1. Add this near the top of Cell 9 (before the rest of the pipeline):

```python
# Change this to choose which process to unlearn (P0-P9)
target_process = "P3"   # ← change to any P0, P1, ..., P9
```

2. Or use an input widget for Jupyter:

```python
from ipywidgets import Dropdown, interact

target_process = Dropdown(
    options=[f"P{i}" for i in range(10)],
    value="P3",
    description="Unlearn:",
)
display(target_process)
# Use target_process.value when running the rest of the cell
```

Then replace the line `target_process="P3"` with `target_process` (or `target_process.value` if using the widget).
"# Verifiable_process_unlearning" 
