"""
CSV process logging for analysis runs and uploads.
"""

import csv
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

LOG_DIR = Path("logs")
LOG_FILE_NAME = "process_logs.csv"

CSV_COLUMNS = [
    "timestamp",
    "run_id",
    "event_type",
    "uploaded_file_name",
    "uploaded_file_extension",
    "stored_file_log_name",
    "question",
    "selected_llm_model",
    "embedding_model",
    "reranker_model",
    "semantic_threshold_mode",
    "naive_chunk_size",
    "naive_chunk_overlap",
    "raw_top_k",
    "final_top_n",
    "child_max_words",
    "child_overlap_words",
    "naive_context_max_chunks",
    "full_document_tokens",
    "naive_context_tokens",
    "engineered_context_tokens",
    "context_size_difference",
    "context_change_percent",
    "context_reduction_percent",
    "naive_estimated_input_tokens",
    "naive_estimated_output_tokens",
    "naive_estimated_total_tokens",
    "engineered_estimated_input_tokens",
    "engineered_estimated_output_tokens",
    "engineered_estimated_total_tokens",
    "document_processing_seconds",
    "chunking_seconds",
    "embedding_seconds",
    "vector_search_seconds",
    "reranking_seconds",
    "naive_llm_seconds",
    "engineered_llm_seconds",
    "total_processing_seconds",
    "status",
    "error_message",
]


def ensure_log_dir():
    """Create the logs directory if it does not exist."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_log_file_path():
    """Return the path to the process logs CSV file."""

    ensure_log_dir()
    return str(LOG_DIR / LOG_FILE_NAME)


def get_current_timestamp():
    """Return an ISO-style timestamp string for log rows."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_uploaded_file_log_name(original_filename):
    """
    Build a timestamped log-friendly file name from the original upload name.

    Example: report.pdf -> report_20260529_142210.pdf
    """

    path = Path(original_filename)
    stem = path.stem or "upload"
    extension = path.suffix or ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{timestamp}{extension}"


def _ensure_log_file_with_headers():
    """Create the CSV file with headers if missing."""

    ensure_log_dir()
    log_path = get_log_file_path()
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def log_process_event(event_data):
    """
    Append one process event row to logs/process_logs.csv.
    """

    _ensure_log_file_with_headers()
    row = {column: event_data.get(column, "") for column in CSV_COLUMNS}
    if not row.get("timestamp"):
        row["timestamp"] = get_current_timestamp()

    with open(get_log_file_path(), "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def read_process_logs():
    """
    Read all process log rows as a pandas DataFrame.
    """

    _ensure_log_file_with_headers()
    log_path = get_log_file_path()
    if os.path.getsize(log_path) == 0:
        return pd.DataFrame(columns=CSV_COLUMNS)

    return pd.read_csv(log_path)
