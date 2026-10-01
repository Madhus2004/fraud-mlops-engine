# Fraud Detection MLOps Engine

A real-time credit card fraud scoring system with drift monitoring, feedback-driven retraining, and a safe model promotion gate.

<img width="1042" height="622" alt="image" src="https://github.com/user-attachments/assets/2a9a3c91-cacd-4886-a1b5-d6c8d910ca6c" />



## What it does
- Scores each transaction with an XGBoost model through a FastAPI service.
- Logs every prediction to Postgres and attaches the true label when it arrives later.
- Detects data drift (KS test and PSI) between live traffic and training data.
- Retrains a challenger model locally and promotes it only if it beats the champion on an unseen holdout set.
- Updates the live API with no redeploy (hot reload from a model registry).
- Tracks every experiment in MLflow.

## Architecture
Two sides share one database and one model registry.

- **Cloud (Render):** API service and dashboard. They serve predictions and show logs.
- **Local (your laptop):** simulator, drift check, retraining worker, MLflow.
- **Shared:** Neon Postgres (logs) and Hugging Face Hub (models).

Life of a transaction: send, score, flag or approve, log, feedback attaches the true label to the same row.

Life of a model: drift or new labels trigger retraining, the challenger competes with the champion, a winner is published to the registry, and the API reloads it.

## Design decisions
- Serving and training are separate, so the small cloud instance never trains.
- Raw features are stored everywhere. The scaler is saved inside each model's manifest, so drift checks and retraining stay consistent.
- Data is split into train, stream, validation and holdout. The threshold is tuned on validation and the gate uses the untouched holdout.
- A candidate is promoted only if its holdout PR-AUC is strictly higher than the champion's. Every candidate is archived.

## Tech stack
Python 3.11, XGBoost, pandas, NumPy, scikit-learn, SciPy, FastAPI, Uvicorn, Streamlit, Plotly, SQLAlchemy, Neon Postgres, Hugging Face Hub, MLflow, Docker, Render, GitHub Actions, pytest, ruff.

## Project structure
- `core/` shared code: feature order and scaling, database layer, model artifacts
- `serving/` API and dashboard
- `training/` data split, training, drift, retraining, gate, worker
- `ops/` traffic and feedback simulator
- `tests/` unit and API tests

## Run locally
Needs the Kaggle `creditcard.csv` file placed in `data/raw/`.

```bash
conda create -n fraud-mlops python=3.11 -y
conda activate fraud-mlops
python -m pip install -r requirements-dev.txt
```

Create a `.env` file (never commit it):
```
DATABASE_URL=your Neon connection string
HF_REPO_ID=your-username/fraud-engine-models
HF_TOKEN=your Hugging Face write token
ADMIN_TOKEN=any long random password
API_URL=your API address
```
For a fully local test, leave `DATABASE_URL` and `HF_REPO_ID` out and use `http://127.0.0.1:8000` as `API_URL`.

Prepare data and train the first model:
```bash
python -m training.make_dataset
python -m training.train
python -m pytest -q
```

Start the services (separate terminals):
```bash
python -m uvicorn serving.main:app --port 8000
python -m streamlit run serving/dashboard.py
```

Run the demo:
```bash
python -m ops.simulator --batches 3 --batch-size 20
python -m ops.simulator --batches 5 --batch-size 40 --drift
python -m training.worker --once
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```
Use `--force` with the worker to retrain regardless of triggers.

## Deploy
- Database: Neon Postgres.
- Models: a private Hugging Face model repo.
- Render: two Docker web services from this repo, one for the API and one for the dashboard. Set `DATABASE_URL`, `HF_REPO_ID`, `HF_TOKEN` (read) and `ADMIN_TOKEN` on both, and set the dashboard's `API_URL` to the API service address.
- Free instances sleep when idle, so the first request can take about a minute.

## Notes
- The simulator sends labels right away. Real chargebacks arrive days later.
- The gate often rejects a retrained model when the champion is already strong. That is the intended safety behavior.
