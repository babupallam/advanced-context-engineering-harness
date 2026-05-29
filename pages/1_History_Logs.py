"""
History Logs page — view and export process_logs.csv.
"""

import streamlit as st
import pandas as pd

from src.evaluation.process_logger import read_process_logs

st.set_page_config(
    page_title="History Logs",
    page_icon="📊",
    layout="wide",
)

st.title("History Logs")

logs_df = read_process_logs()

if logs_df.empty:
    st.info("No process logs yet. Run an analysis from the main app to create log entries.")
    st.stop()

total_runs = len(logs_df)
successful_runs = len(logs_df[logs_df["status"] == "success"]) if "status" in logs_df else 0
failed_runs = len(logs_df[logs_df["status"] == "failed"]) if "status" in logs_df else 0

metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5, metric_col_6 = st.columns(6)

with metric_col_1:
    st.metric("Total Runs", total_runs)

with metric_col_2:
    st.metric("Successful Runs", successful_runs)

with metric_col_3:
    st.metric("Failed Runs", failed_runs)

with metric_col_4:
    avg_total = logs_df["total_processing_seconds"].mean() if "total_processing_seconds" in logs_df else 0
    st.metric("Avg Total Processing Time (s)", round(float(avg_total or 0), 4))

with metric_col_5:
    avg_naive = logs_df["naive_llm_seconds"].mean() if "naive_llm_seconds" in logs_df else 0
    st.metric("Avg Naive LLM Time (s)", round(float(avg_naive or 0), 4))

with metric_col_6:
    avg_engineered = (
        logs_df["engineered_llm_seconds"].mean()
        if "engineered_llm_seconds" in logs_df
        else 0
    )
    st.metric("Avg Engineered LLM Time (s)", round(float(avg_engineered or 0), 4))

st.divider()

filtered_df = logs_df.copy()

status_options = ["All"]
if "status" in filtered_df.columns:
    status_options += sorted(
        [value for value in filtered_df["status"].dropna().unique().tolist() if value != ""]
    )

model_options = ["All"]
if "selected_llm_model" in filtered_df.columns:
    model_options += sorted(
        [
            value
            for value in filtered_df["selected_llm_model"].dropna().unique().tolist()
            if value != ""
        ]
    )

extension_options = ["All"]
if "uploaded_file_extension" in filtered_df.columns:
    extension_options += sorted(
        [
            value
            for value in filtered_df["uploaded_file_extension"].dropna().unique().tolist()
            if value != ""
        ]
    )

filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns(4)

with filter_col_1:
    status_filter = st.selectbox("Status filter", status_options)

with filter_col_2:
    model_filter = st.selectbox("Model filter", model_options)

with filter_col_3:
    extension_filter = st.selectbox("File extension filter", extension_options)

with filter_col_4:
    if "timestamp" in filtered_df.columns:
        filtered_df["_log_date"] = pd.to_datetime(
            filtered_df["timestamp"], errors="coerce"
        ).dt.date
        available_dates = sorted(
            [value for value in filtered_df["_log_date"].dropna().unique().tolist()],
            reverse=True,
        )
        date_filter = st.selectbox(
            "Date filter",
            ["All"] + [str(value) for value in available_dates],
        )
    else:
        date_filter = "All"

if status_filter != "All" and "status" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["status"] == status_filter]

if model_filter != "All" and "selected_llm_model" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["selected_llm_model"] == model_filter]

if extension_filter != "All" and "uploaded_file_extension" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["uploaded_file_extension"] == extension_filter]

if date_filter != "All" and "_log_date" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["_log_date"].astype(str) == date_filter]

if "_log_date" in filtered_df.columns:
    filtered_df = filtered_df.drop(columns=["_log_date"])

st.subheader("Process Log Entries")
st.dataframe(filtered_df, width="stretch", hide_index=True)

csv_data = filtered_df.to_csv(index=False)
st.download_button(
    "Download Logs CSV",
    data=csv_data,
    file_name="process_logs.csv",
    mime="text/csv",
)
