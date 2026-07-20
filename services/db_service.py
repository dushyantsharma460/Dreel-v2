"""
Database Service — optional PostgreSQL mirror

Additive only: data/*.csv remains the source of truth for the Jupyter
notebooks, and uploads/ + static/reels/ remain the source of truth for
files on disk. This module mirrors the same rows/files into Postgres
(audio_features, image_features, engagement_ratings, reel_files) purely
so the project's data is also browsable in a normal SQL tool such as
pgAdmin4.

Point the app at any existing database by editing PG_* / DATABASE_URL
in .env (see config.py) — nothing here needs to change.

If Postgres is unreachable, every write function logs a warning and
returns without raising, so the reel pipeline never depends on the
database being up.
"""

import mimetypes
import os
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    create_engine,
    text,
)
from sqlalchemy.exc import SQLAlchemyError

from config import DATABASE_URL

metadata = MetaData()

audio_features_table = Table(
    "audio_features",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True)),
    Column("reel_id", String, index=True),
    Column("image_count", Integer),
    Column("sync_mode", String),
    Column("duration_sec", Float),
    Column("tempo_bpm", Float),
    Column("beat_count", Integer),
    Column("avg_energy", Float),
    Column("avg_slide_duration", Float),
    Column("slide_durations", String),
)

image_features_table = Table(
    "image_features",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True)),
    Column("reel_id", String, index=True),
    Column("filename", String),
    Column("sort_rank", Integer),
    Column("blur_variance", Float),
    Column("brightness", Float),
    Column("contrast", Float),
    Column("quality_score", Float),
    Column("quality_label", String),
)

engagement_ratings_table = Table(
    "engagement_ratings",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True)),
    Column("reel_id", String, unique=True, index=True),
    Column("rating", Integer),
    Column("source", String),
)

# Raw file bytes for uploaded images/audio and the final rendered video.
# file_type is "image" / "audio" / "video". Large blobs in Postgres are
# heavier than plain filesystem storage (bigger DB, slower backups) —
# kept here because the project explicitly wants files queryable in
# pgAdmin4 too, not just their extracted feature rows.
reel_files_table = Table(
    "reel_files",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("reel_id", String, index=True),
    Column("file_type", String, index=True),
    Column("filename", String),
    Column("content_type", String),
    Column("file_size", Integer),
    Column("file_data", LargeBinary),
    Column("created_at", DateTime(timezone=True)),
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def init_db() -> bool:
    """Create the mirror tables if they don't already exist."""

    try:
        metadata.create_all(get_engine())
        print("[DB] Postgres mirror tables ready.")
        return True
    except SQLAlchemyError as error:
        print(f"[DB][WARN] Could not initialize Postgres mirror: {error}")
        return False


def mirror_audio_features(row: dict) -> None:
    try:
        with get_engine().begin() as conn:
            conn.execute(audio_features_table.insert(), row)
    except SQLAlchemyError as error:
        print(f"[DB][WARN] audio_features mirror insert failed: {error}")


def mirror_image_features(rows: list[dict]) -> None:
    if not rows:
        return

    try:
        with get_engine().begin() as conn:
            conn.execute(image_features_table.insert(), rows)
    except SQLAlchemyError as error:
        print(f"[DB][WARN] image_features mirror insert failed: {error}")


def mirror_rating(row: dict) -> None:
    """Insert-or-replace by reel_id, mirroring the CSV's dedupe behavior."""

    try:
        with get_engine().begin() as conn:
            conn.execute(
                text("DELETE FROM engagement_ratings WHERE reel_id = :reel_id"),
                {"reel_id": row["reel_id"]},
            )
            conn.execute(engagement_ratings_table.insert(), row)
    except SQLAlchemyError as error:
        print(f"[DB][WARN] engagement_ratings mirror insert failed: {error}")


def store_reel_file(reel_id: str, file_type: str, filename: str, file_path: str) -> None:
    """
    Read a file from disk and store its bytes in Postgres, replacing any
    prior copy with the same reel_id/file_type/filename (so re-processing
    a folder — e.g. a retried reel — doesn't pile up duplicate blobs).
    """

    try:
        with open(file_path, "rb") as source_file:
            data = source_file.read()
    except OSError as error:
        print(f"[DB][WARN] Could not read {file_path} for DB storage: {error}")
        return

    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    try:
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    "DELETE FROM reel_files "
                    "WHERE reel_id = :reel_id AND file_type = :file_type AND filename = :filename"
                ),
                {"reel_id": reel_id, "file_type": file_type, "filename": filename},
            )
            conn.execute(reel_files_table.insert(), {
                "reel_id": reel_id,
                "file_type": file_type,
                "filename": filename,
                "content_type": content_type,
                "file_size": len(data),
                "file_data": data,
                "created_at": datetime.now(timezone.utc),
            })
    except SQLAlchemyError as error:
        print(f"[DB][WARN] reel_files insert failed for {filename}: {error}")


def store_reel_files(reel_id: str, file_type: str, upload_dir: str, filenames: list[str]) -> None:
    for filename in filenames:
        store_reel_file(reel_id, file_type, filename, os.path.join(upload_dir, filename))


def delete_reel_records(reel_id: str) -> None:
    """Best-effort removal of every mirrored row for one reel_id."""

    try:
        with get_engine().begin() as conn:
            for table in (
                reel_files_table,
                audio_features_table,
                image_features_table,
                engagement_ratings_table,
            ):
                conn.execute(table.delete().where(table.c.reel_id == reel_id))
    except SQLAlchemyError as error:
        print(f"[DB][WARN] delete_reel_records failed for {reel_id}: {error}")


def get_reel_file(reel_id: str, file_type: str) -> dict | None:
    """Fetch one stored file's bytes + metadata, or None if not mirrored."""

    try:
        with get_engine().connect() as conn:
            row = conn.execute(
                text(
                    "SELECT filename, content_type, file_size, file_data FROM reel_files "
                    "WHERE reel_id = :reel_id AND file_type = :file_type "
                    "ORDER BY id DESC LIMIT 1"
                ),
                {"reel_id": reel_id, "file_type": file_type},
            ).mappings().first()
            return dict(row) if row else None
    except SQLAlchemyError as error:
        print(f"[DB][WARN] reel_files lookup failed for {reel_id}/{file_type}: {error}")
        return None
