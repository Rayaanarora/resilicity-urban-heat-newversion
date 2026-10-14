"""SQLite persistence for projects (stdlib sqlite3, no ORM).

Database path: PROJECTS_DB env var, default backend/data/projects.sqlite3. Uploaded photos and generated images
live next to it in projects/<project_id>/.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

BACKEND = Path(__file__).resolve().parent.parent
DEFAULT_DB = BACKEND / "data" / "projects.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id              TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    name            TEXT NOT NULL,
    lat             REAL,
    lon             REAL,
    cell_id         TEXT,
    road_width_m    REAL,
    climate_zone    TEXT,
    budget_lakh     REAL,
    photo_path      TEXT NOT NULL,
    photo_sha256    TEXT NOT NULL,
    hotspot_json    TEXT,
    analysis_json   TEXT,
    plan_json       TEXT,
    thermal_json    TEXT,
    redesign_status TEXT NOT NULL DEFAULT 'none',
    redesign_path   TEXT,
    redesign_error  TEXT,
    redesign_meta_json TEXT
);
CREATE TABLE IF NOT EXISTS project_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    started_at  REAL NOT NULL,
    finished_at REAL,
    status      TEXT NOT NULL,
    detail_json TEXT
);
CREATE TABLE IF NOT EXISTS measurements (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    created_at   REAL NOT NULL,
    payload_json TEXT NOT NULL,
    result_json  TEXT NOT NULL
);
"""

JSON_COLUMNS = ("hotspot_json", "analysis_json", "plan_json", "thermal_json", "redesign_meta_json")


def db_path() -> Path:
    return Path(os.environ.get("PROJECTS_DB", str(DEFAULT_DB)))


def files_dir(project_id: str) -> Path:
    d = db_path().parent / "projects" / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _decode(row: sqlite3.Row) -> Dict[str, Any]:
    out = dict(row)
    for col in JSON_COLUMNS:
        if col in out:
            out[col.removesuffix("_json")] = json.loads(out.pop(col)) if out[col] else None
    return out


def insert_project(record: Dict[str, Any]) -> None:
    now = time.time()
    rec = {**record, "created_at": now, "updated_at": now}
    for col in JSON_COLUMNS:
        key = col.removesuffix("_json")
        if key in rec:
            rec[col] = json.dumps(rec.pop(key))
    cols = ", ".join(rec)
    with connect() as c:
        c.execute(f"INSERT INTO projects ({cols}) VALUES ({', '.join('?' * len(rec))})", list(rec.values()))


def update_project(project_id: str, **fields: Any) -> None:
    sets, vals = [], []
    for key, val in fields.items():
        col = f"{key}_json" if f"{key}_json" in JSON_COLUMNS else key
        sets.append(f"{col} = ?")
        vals.append(json.dumps(val) if col in JSON_COLUMNS else val)
    sets.append("updated_at = ?")
    vals.append(time.time())
    with connect() as c:
        c.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", [*vals, project_id])


def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    with connect() as c:
        row = c.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return _decode(row) if row else None


def list_projects() -> List[Dict[str, Any]]:
    with connect() as c:
        rows = c.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
    return [_decode(r) for r in rows]


def delete_project(project_id: str) -> bool:
    with connect() as c:
        cur = c.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    return cur.rowcount > 0


def log_run(project_id: str, kind: str, status: str, detail: Optional[Dict[str, Any]] = None,
            started_at: Optional[float] = None) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO project_runs (project_id, kind, started_at, finished_at, status, detail_json) VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, kind, started_at or time.time(), time.time(), status, json.dumps(detail or {})),
        )


def list_runs(project_id: str) -> List[Dict[str, Any]]:
    with connect() as c:
        rows = c.execute("SELECT * FROM project_runs WHERE project_id = ? ORDER BY id", (project_id,)).fetchall()
    return [{**dict(r), "detail": json.loads(r["detail_json"] or "{}")} for r in rows]


def add_measurement(project_id: str, payload: Dict[str, Any], result: Dict[str, Any]) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO measurements (project_id, created_at, payload_json, result_json) VALUES (?, ?, ?, ?)",
            (project_id, time.time(), json.dumps(payload), json.dumps(result)),
        )
    return int(cur.lastrowid)


def list_measurements(project_id: str) -> List[Dict[str, Any]]:
    with connect() as c:
        rows = c.execute("SELECT * FROM measurements WHERE project_id = ? ORDER BY id", (project_id,)).fetchall()
    return [{"id": r["id"], "created_at": r["created_at"], "payload": json.loads(r["payload_json"]),
             "result": json.loads(r["result_json"])} for r in rows]
