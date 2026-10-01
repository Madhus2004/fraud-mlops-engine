from core.features import FEATURE_COLUMNS
from tests.conftest import make_frame


def _payload():
    return {c: float(v) for c, v in make_frame(1, seed=3)[FEATURE_COLUMNS].iloc[0].items()}


def test_health(api_client):
    body = api_client.get("/health").json()
    assert body["active_model_version"] == "v1" and body["decision_threshold"] == 0.5


def test_predict_logs_and_feedback_roundtrip(api_client):
    r = api_client.post("/predict", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0 and body["model_version"] == "v1"

    ok = api_client.post("/feedback", json={"txn_id": body["txn_id"], "actual_label": 1})
    assert ok.status_code == 200  # regression: used to always 404

    missing = api_client.post("/feedback", json={"txn_id": "nope", "actual_label": 1})
    assert missing.status_code == 404


def test_logged_features_are_raw(api_client):
    from core.db import expand_features, fetch_logs
    payload = _payload()
    api_client.post("/predict", json=payload)
    logged = expand_features(fetch_logs()).iloc[0]
    assert logged["Amount"] == payload["Amount"] and logged["Time"] == payload["Time"]


def test_predict_rejects_bad_payload(api_client):
    assert api_client.post("/predict", json={"Amount": 1.0}).status_code == 422


def test_admin_reload_requires_token(api_client):
    assert api_client.post("/admin/reload").status_code == 401
    assert api_client.post("/admin/reload", headers={"X-Admin-Token": "secret"}).status_code == 200
