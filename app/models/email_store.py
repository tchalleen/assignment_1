import os
import json
import tempfile
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import asdict
from app.dataclasses import Email


class EmailStore:
    """Simple JSON-backed store for emails with optional ground truth topic"""

    def __init__(self):
        self._emails: Dict[str, Dict[str, Any]] = {}
        self._file_path = self._get_store_file_path()
        self._load()

    def _get_store_file_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "emails.json"
        )

    def _load(self) -> None:
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._emails = data.get("emails", {})
            except Exception as e:
                # If corrupted, start fresh
                self._emails = {}
        else:
            self._emails = {}

    def _save(self) -> None:
        data_dir = os.path.dirname(self._file_path)
        os.makedirs(data_dir, exist_ok=True)

        payload = {"emails": self._emails}

        with tempfile.NamedTemporaryFile("w", delete=False, dir=data_dir, encoding="utf-8") as tmp:
            json.dump(payload, tmp, indent=2, ensure_ascii=False)
            tmp.write("\n")
            tmp_path = tmp.name

        os.replace(tmp_path, self._file_path)

    def add_email(self, email: Email, ground_truth_topic: Optional[str] = None) -> str:
        email_id = str(uuid.uuid4())
        record = {
            "id": email_id,
            "subject": email.subject,
            "body": email.body,
            "ground_truth_topic": ground_truth_topic,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._emails[email_id] = record
        self._save()
        return email_id

    def get_all_emails(self) -> List[Dict[str, Any]]:
        return list(self._emails.values())

    def get_emails_with_ground_truth(self) -> List[Dict[str, Any]]:
        return [e for e in self._emails.values() if e.get("ground_truth_topic")]

    def clear(self) -> None:
        self._emails = {}
        self._save()
