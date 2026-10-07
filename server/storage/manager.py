from __future__ import annotations

import io
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

STORAGE_DIR = Path(__file__).resolve().parent
LAST_JSON_PATH = STORAGE_DIR / "last_slide.json"
LAST_IMAGE_PATH = STORAGE_DIR / "last_slide_img.jpg"
LAST_TEXT_PATH = STORAGE_DIR / "last_slide_text.txt"
LAST_CLIP_PATH = STORAGE_DIR / "last_slide_clip.npy"
SLIDES_LOG_PATH = STORAGE_DIR / "slides_log.json"
SQLITE_PATH = STORAGE_DIR / "slidescribe.db"
DEFAULT_SESSION_ID = "default"


@dataclass
class SlideState:
    summary: Optional[Dict[str, Any]] = None
    image_path: Optional[Path] = None
    text: Optional[str] = None
    clip_vector: Optional[np.ndarray] = None


def _ensure_storage_dir() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_storage_dir()
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS session_states (
            session_id TEXT PRIMARY KEY,
            summary_json TEXT,
            ocr_text TEXT,
            clip_vector BLOB,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS slides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            slide_number INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            ocr_text TEXT,
            text_similarity REAL,
            clip_cosine REAL,
            UNIQUE(session_id, slide_number),
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS qa_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            slide_number INTEGER,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            relevance REAL,
            matched_slides_json TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        );
        """
    )
    conn.commit()


def _now() -> str:
    return datetime.utcnow().isoformat()


def _normalize_session_id(session_id: Optional[str]) -> str:
    value = (session_id or DEFAULT_SESSION_ID).strip()
    return value or DEFAULT_SESSION_ID


def ensure_session(session_id: Optional[str] = None) -> str:
    """Create the session row if needed and return the normalized session id."""
    normalized = _normalize_session_id(session_id)
    timestamp = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions(session_id, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at
            """,
            (normalized, timestamp, timestamp),
        )
        conn.commit()
    return normalized


def _load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _serialize_vector(vector: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, vector)
    return buffer.getvalue()


def _deserialize_vector(payload: Optional[bytes]) -> Optional[np.ndarray]:
    if payload is None:
        return None
    try:
        return np.load(io.BytesIO(payload))
    except ValueError:
        return None


def load_last_state(session_id: Optional[str] = None) -> SlideState:
    """Load the last processed slide information from storage."""
    normalized = _normalize_session_id(session_id)
    _ensure_storage_dir()

    if normalized != DEFAULT_SESSION_ID or SQLITE_PATH.exists():
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT summary_json, ocr_text, clip_vector
                FROM session_states
                WHERE session_id = ?
                """,
                (normalized,),
            ).fetchone()
        if row is not None:
            summary = json.loads(row["summary_json"]) if row["summary_json"] else None
            return SlideState(
                summary=summary,
                text=row["ocr_text"],
                clip_vector=_deserialize_vector(row["clip_vector"]),
            )

    summary: Optional[Dict[str, Any]] = None
    image_path: Optional[Path] = None
    text: Optional[str] = None
    clip_vector: Optional[np.ndarray] = None

    if LAST_JSON_PATH.exists():
        summary = _load_json_file(LAST_JSON_PATH, None)

    if LAST_TEXT_PATH.exists():
        text = LAST_TEXT_PATH.read_text(encoding="utf-8")

    if LAST_CLIP_PATH.exists():
        try:
            clip_vector = np.load(LAST_CLIP_PATH)
        except ValueError:
            clip_vector = None

    if LAST_IMAGE_PATH.exists():
        image_path = LAST_IMAGE_PATH

    return SlideState(
        summary=summary,
        image_path=image_path,
        text=text,
        clip_vector=clip_vector,
    )


def save_last_state(
    image: np.ndarray,
    *,
    summary: Optional[Dict[str, Any]],
    text: str,
    clip_vector: np.ndarray,
    session_id: Optional[str] = None,
) -> None:
    """Persist the latest slide artifacts."""
    normalized = ensure_session(session_id)
    timestamp = _now()
    summary_json = json.dumps(summary, ensure_ascii=False) if summary is not None else None

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO session_states(session_id, summary_json, ocr_text, clip_vector, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                summary_json=excluded.summary_json,
                ocr_text=excluded.ocr_text,
                clip_vector=excluded.clip_vector,
                updated_at=excluded.updated_at
            """,
            (normalized, summary_json, text, _serialize_vector(clip_vector), timestamp),
        )
        conn.commit()

    if normalized != DEFAULT_SESSION_ID:
        return

    _ensure_storage_dir()
    LAST_TEXT_PATH.write_text(text, encoding="utf-8")
    np.save(LAST_CLIP_PATH, clip_vector)

    if summary is not None:
        LAST_JSON_PATH.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    success = cv2.imwrite(str(LAST_IMAGE_PATH), image)
    if not success:
        raise RuntimeError(f"Failed to save slide image to {LAST_IMAGE_PATH}")


def load_slide_history(
    limit: Optional[int] = None,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the stored slide history, optionally truncated to the last [limit] entries."""
    normalized = _normalize_session_id(session_id)
    _ensure_storage_dir()

    if normalized != DEFAULT_SESSION_ID or SQLITE_PATH.exists():
        query = """
            SELECT slide_number, timestamp, summary_json, ocr_text, text_similarity, clip_cosine
            FROM slides
            WHERE session_id = ?
            ORDER BY slide_number ASC
        """
        params: tuple[Any, ...] = (normalized,)
        if limit is not None and limit > 0:
            query = """
                SELECT * FROM (
                    SELECT slide_number, timestamp, summary_json, ocr_text, text_similarity, clip_cosine
                    FROM slides
                    WHERE session_id = ?
                    ORDER BY slide_number DESC
                    LIMIT ?
                )
                ORDER BY slide_number ASC
            """
            params = (normalized, limit)
        with _connect() as conn:
            rows = conn.execute(query, params).fetchall()
        if rows:
            return [_row_to_slide_entry(row) for row in rows]
        if normalized != DEFAULT_SESSION_ID:
            return []

    history: List[Dict[str, Any]] = _load_json_file(SLIDES_LOG_PATH, default=[])
    if limit is not None and limit > 0:
        return history[-limit:]
    return history


def _row_to_slide_entry(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "slide_number": row["slide_number"],
        "timestamp": row["timestamp"],
        "summary": json.loads(row["summary_json"]),
        "ocr_text": row["ocr_text"],
        "metrics": {
            "text_similarity": row["text_similarity"],
            "clip_cosine": row["clip_cosine"],
        },
    }


def append_slide_history(
    entry: Dict[str, Any],
    session_id: Optional[str] = None,
) -> None:
    """Append a new slide entry to the history log."""
    normalized = ensure_session(session_id)
    metrics = entry.get("metrics", {}) or {}
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO slides(
                session_id,
                slide_number,
                timestamp,
                summary_json,
                ocr_text,
                text_similarity,
                clip_cosine
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized,
                entry["slide_number"],
                entry.get("timestamp") or _now(),
                json.dumps(entry.get("summary", {}), ensure_ascii=False),
                entry.get("ocr_text"),
                metrics.get("text_similarity"),
                metrics.get("clip_cosine"),
            ),
        )
        conn.commit()

    if normalized != DEFAULT_SESSION_ID:
        return

    _ensure_storage_dir()
    history = _load_json_file(SLIDES_LOG_PATH, default=[])
    history.append(entry)
    SLIDES_LOG_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def log_qa_turn(
    *,
    session_id: Optional[str],
    slide_number: Optional[int],
    question: str,
    answer: str,
    relevance: Optional[float],
    matched_slides: List[int],
) -> None:
    """Persist a question-answer turn for session replay/debugging."""
    normalized = ensure_session(session_id)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO qa_turns(
                session_id,
                slide_number,
                question,
                answer,
                relevance,
                matched_slides_json,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized,
                slide_number,
                question,
                answer,
                relevance,
                json.dumps(matched_slides),
                _now(),
            ),
        )
        conn.commit()


def reset_slide_history(session_id: Optional[str] = None) -> None:
    """Remove the stored slide history."""
    normalized = _normalize_session_id(session_id)
    if SQLITE_PATH.exists():
        with _connect() as conn:
            conn.execute("DELETE FROM slides WHERE session_id = ?", (normalized,))
            conn.execute("DELETE FROM qa_turns WHERE session_id = ?", (normalized,))
            conn.execute("DELETE FROM session_states WHERE session_id = ?", (normalized,))
            conn.commit()

    if normalized == DEFAULT_SESSION_ID and SLIDES_LOG_PATH.exists():
        SLIDES_LOG_PATH.unlink()
