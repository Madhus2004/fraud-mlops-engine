# Fraud Detection MLOps Engine

Real-time credit-card fraud scoring with a **cloud serving plane** and a **local training plane**.
Heavy work (drift analysis, retraining, evaluation) never touches the 512 MB Render instance.

```
Simulator/Users ─► FastAPI (Render) ─► Postgres (Neon) ◄──── Local worker
                       ▲   │                                   │ drift → retrain → gate
                       │   └─ Streamlit (test + logs)          │ MLflow tracking
                       └──── POST /admin/reload ◄── HF Hub ◄───┘ (model registry)
```

| Path | Role |
|---|---|
| `core/` | Shared: feature order + scaling, DB layer, artifact store/registry |
| `serving/` | FastAPI (`/predict /feedback /health /admin/reload`) + 2-tab Streamlit dashboard |
| `training/` | Dataset split, train v1, KS+PSI drift, retrain, champion/challenger gate, worker |
| `ops/` | Traffic + delayed-feedback simulator |
| `tests/` | API contract, drift, gate, threshold, scaling |

## Design decisions
- **RAW features everywhere** (DB, reference data). The scaler is stored in each model's `manifest.json`, so drift checks and retraining can never mix scaled and unscaled data.
- **Three-way discipline:** threshold tuned on `validation`, promotion gate on untouched `holdout`, live traffic from a separate `stream_pool`.
- **Champion/challenger gate:** a retrained model is promoted only if its holdout PR-AUC beats the champion. Every candidate is archived in `artifacts/versions/`.
- **Hot reload:** API loads the model once at startup, cached; the worker calls `/admin/reload` after promotion (no redeploy).
- **Light serving image:** no scikit-learn/scipy/mlflow; uses the XGBoost `Booster` and numpy for scaling.

## Local setup
```bash
pip install -r requirements-dev.txt
cp .env.example .env            # for local-only dev you can skip DATABASE_URL (SQLite fallback)
# put creditcard.csv in data/raw/
python -m training.make_dataset
python -m training.train        # creates v1 in artifacts/active
pytest -q
```

## Run the demo (three terminals)
```bash
uvicorn serving.main:app --port 8000
streamlit run serving/dashboard.py
python -m ops.simulator --batches 5 --batch-size 40            # normal traffic + feedback
python -m ops.simulator --batches 5 --batch-size 40 --drift    # inject drift
python -m training.worker --once                               # drift -> retrain -> gate -> publish
mlflow ui --backend-store-uri sqlite:///mlflow.db                                                       # inspect runs
```
`--force` retrains regardless of triggers. Triggers: drift share ≥ 30%, ≥ 50 new labels, or manual.

## Deploy
1. **Neon:** create a Postgres DB, copy `DATABASE_URL`.
2. **Hugging Face:** create a model repo, a write token (worker) and a read token (Render).
3. Run `python -m training.train`, then push v1 once: `python -c "from core.artifacts import *; push_to_hub('artifacts/versions/v1','v1')"` (with `HF_REPO_ID`/`HF_TOKEN` set).
4. **Render** (Docker web service): set `DATABASE_URL`, `HF_REPO_ID`, `HF_TOKEN` (read), `ADMIN_TOKEN`, and `API_URL=http://127.0.0.1:8000`.
5. Local worker `.env`: same `DATABASE_URL`, `HF_*` (write), `ADMIN_TOKEN`, and `API_URL=https://<your-service>.onrender.com`.

## Notes
- Free Render instances sleep; the first request after idle can take ~50 s.
- The Dockerfile copies `artifacts/` as a fallback model if the Hub is unreachable.
