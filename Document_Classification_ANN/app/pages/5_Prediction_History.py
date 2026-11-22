"""Prediction history (metadata only; document contents are never stored)."""
import streamlit as st

from components.ui import config, metric_card, setup_page
from src import history
from src.config import resolve

setup_page("Prediction History", "Local log of predictions. Uploaded document contents are not stored.")
cfg = config()
path = resolve(cfg, "history_file")
df = history.load_history(path)

if df.empty:
    st.info("No predictions recorded yet. Classify a document to populate this page.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    metric_card("Predictions", f"{len(df):,}")
with c2:
    metric_card("Average confidence", f"{df['confidence'].mean()*100:.1f}%")
with c3:
    metric_card("Most frequent category", str(df["predicted_category"].mode().iloc[0]))

st.dataframe(df.iloc[::-1], hide_index=True, use_container_width=True)
st.bar_chart(df["predicted_category"].value_counts(), color="#3949ab")
st.download_button("Download history as CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name="prediction_history.csv", mime="text/csv")

st.divider()
confirm = st.checkbox("I want to permanently delete the prediction history")
if st.button("Clear history", disabled=not confirm):
    history.clear_history(path)
    st.success("Prediction history cleared.")
    st.rerun()
