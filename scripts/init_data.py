#!/usr/bin/env python3

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ensure_entity_merge_rules(path: Path, *, reset: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not reset:
        data = _load_json(path)
        if isinstance(data, dict) and isinstance(data.get("merge_rules"), dict):
            if "last_updated" not in data:
                data["last_updated"] = _utc_now_iso()
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        bak = path.with_suffix(path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        bak.write_text(path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    payload = {"merge_rules": {}, "last_updated": _utc_now_iso()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _delete_if_exists(p: Path) -> bool:
    try:
        if p.exists():
            p.unlink()
            return True
    except Exception:
        return False
    return False


def _reset_sqlite_files(data_dir: Path) -> int:
    cnt = 0
    for name in ["store.sqlite", "store.sqlite-wal", "store.sqlite-shm"]:
        if _delete_if_exists(data_dir / name):
            cnt += 1
    return cnt


def _reset_tmp_news(data_dir: Path) -> int:
    tmp_dir = data_dir / "tmp"
    removed = 0
    for sub in ["raw_news", "deduped_news"]:
        d = tmp_dir / sub
        if not d.exists():
            continue
        for f in d.glob("*.jsonl"):
            if _delete_if_exists(f):
                removed += 1
    return removed


def _ensure_data_dirs(data_dir: Path) -> None:
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)
    (data_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    (data_dir / "tmp" / "raw_news").mkdir(parents=True, exist_ok=True)
    (data_dir / "tmp" / "deduped_news").mkdir(parents=True, exist_ok=True)
    (data_dir / "projects" / "default" / "runs").mkdir(parents=True, exist_ok=True)


def _init_sqlite_and_exports(project_root: Path) -> None:
    sys.path.insert(0, str(project_root))
    from src.adapters.sqlite.store import get_store

    store = get_store()
    store.export_compat_json_files()


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parent.parent
    config_dir = project_root / "config"
    data_dir = project_root / "data"
    rules_path = config_dir / "entity_merge_rules.json"

    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--reset-tmp", action="store_true")
    ns = ap.parse_args(argv)

    _ensure_data_dirs(data_dir)
    _ensure_entity_merge_rules(rules_path, reset=bool(ns.reset))

    stats = {
        "reset_sqlite_files": 0,
        "reset_tmp_files": 0,
        "entity_merge_rules": str(rules_path),
        "data_dir": str(data_dir),
    }

    if ns.reset:
        stats["reset_sqlite_files"] = _reset_sqlite_files(data_dir)
        stats["reset_tmp_files"] = _reset_tmp_news(data_dir)
    elif ns.reset_tmp:
        stats["reset_tmp_files"] = _reset_tmp_news(data_dir)

    _init_sqlite_and_exports(project_root)

    print(json.dumps({"status": "ok", **stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

