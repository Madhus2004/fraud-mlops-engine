import os

import requests

from core.artifacts import hub_configured, push_to_hub


def publish(version_dir, version: str) -> None:
    if hub_configured() and os.getenv("HF_TOKEN"):
        push_to_hub(version_dir, version)
        print(f"Pushed {version} to Hugging Face Hub.")
    else:
        print("HF_REPO_ID/HF_TOKEN not set; skipping Hub upload (local only).")
    notify_api()


def notify_api() -> None:
    url, token = os.getenv("API_URL"), os.getenv("ADMIN_TOKEN")
    if not url or not token:
        print("API_URL/ADMIN_TOKEN not set; skipping hot reload.")
        return
    try:  # generous timeout: free Render instances need time to wake up
        r = requests.post(f"{url.rstrip('/')}/admin/reload", headers={"X-Admin-Token": token}, timeout=120)
        print("API reload:", r.status_code, r.text)
    except Exception as exc:
        print(f"Could not reload API: {exc}")
