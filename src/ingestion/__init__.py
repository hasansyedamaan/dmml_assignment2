"""Data ingestion for MovieLens 100K and the documented synthetic fallback."""

from src.ingestion.ingest import ingest_data, inject_faults

__all__ = ["ingest_data", "inject_faults"]
