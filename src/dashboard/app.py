import logging
import os

import gradio as gr
import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


TOPIC_LABELS = {
    "Robotics": "robotics",
    "SLMs": "slm",
    "Multimodal Models": "multimodal_reasoning",
}

TABLE_HEADERS = [
    "Model",
    "Predicted Downloads (30d)",
    "Top Quartile Probability (%)",
    "Top Quartile",
]

PLACEHOLDER = "Select a topic"

CSS = """
:root {
    --hf-yellow: #FFD21E;
    --hf-yellow-soft: #FFF6BF;
    --hf-bg: #FAFAF7;
    --hf-card-bg: #FFFFFF;
    --hf-ink: #0F172A;
    --hf-muted: #6B7280;
    --hf-border: #E5E7EB;
    --hf-border-strong: #D1D5DB;
}
/* Hide Gradio's default footer ("Use via API · Built with Gradio · Settings") */
footer { display: none !important; }
html, body, .gradio-container, gradio-app {
    background: var(--hf-bg) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: var(--hf-ink) !important;
}
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 40px 64px !important;
}
/* Override Gradio's default max-width media query */
.gradio-container,
.gradio-container > div,
.gradio-container > div > div,
#component-0,
#component-0 > div {
    max-width: 100% !important;
    width: 100% !important;
}
.content {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
}
.header-text { text-align: center; margin-bottom: 28px; }
.header-text h1 {
    margin: 0;
    color: var(--hf-ink);
    font-size: 2.25rem;
    font-weight: 700;
    letter-spacing: -0.025em;
}
.header-text p {
    color: var(--hf-muted);
    margin: 10px 0 0;
    font-size: 1rem;
}
.metric-card {
    background: var(--hf-card-bg);
    border: 1px solid var(--hf-border);
    border-radius: 16px;
    padding: 28px;
    aspect-ratio: 1 / 1;
    width: 340px;
    margin: 0;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 8px 24px rgba(15, 23, 42, 0.04);
}
.cards-row {
    justify-content: center !important;
    gap: 24px !important;
}
.cards-row > * {
    flex: 0 0 auto !important;
    min-width: 0 !important;
    width: auto !important;
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--hf-muted);
    font-weight: 600;
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--hf-ink);
    line-height: 1.3;
    word-break: break-word;
    overflow: hidden;
}
.metric-sub {
    font-size: 0.875rem;
    color: var(--hf-muted);
    font-weight: 400;
}
.last-updated p {
    color: var(--hf-muted);
    font-size: 0.82rem;
    text-align: right;
    margin: 10px 4px 0 0;
}
/* Controls row: vertically centre the dropdown and refresh button, and match the
   85%-wide centred table so the left edges line up */
.controls-row {
    align-items: center !important;
    justify-content: flex-start !important;
    width: 85% !important;
    margin: 0 auto !important;
}

/* Topic dropdown */
#topic-dropdown, #topic-dropdown > div, #topic-dropdown .wrap, #topic-dropdown .container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 190px !important;
}
#topic-dropdown input {
    background: transparent !important;
    border: 1px solid var(--hf-border-strong) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    width: 100% !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--hf-ink) !important;
}
#topic-dropdown input:focus {
    border-color: var(--hf-yellow) !important;
    box-shadow: 0 0 0 3px rgba(255, 210, 30, 0.25) !important;
    outline: none !important;
}

/* Refresh button */
#refresh-btn button, #refresh-btn {
    background: var(--hf-yellow) !important;
    color: var(--hf-ink) !important;
    border: 1px solid var(--hf-ink) !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 14px 22px !important;
    box-shadow: none !important;
    transition: background 0.15s ease;
}
#refresh-btn button:hover {
    background: #FFC700 !important;
}
/* Table */
#predictions-table {
    width: 85% !important;
    margin: 0 auto !important;
}
.gradio-container .table-wrap, .gradio-container table {
    border: 1px solid var(--hf-border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.gradio-container thead th {
    background: var(--hf-bg) !important;
    color: var(--hf-ink) !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--hf-border) !important;
}
.gradio-container tbody tr:hover td {
    background: rgba(255, 210, 30, 0.08) !important;
}
.gradio-container a {
    color: var(--hf-ink) !important;
    font-weight: 500;
    text-decoration: underline;
    text-decoration-color: var(--hf-yellow);
    text-decoration-thickness: 2px;
    text-underline-offset: 3px;
}
.gradio-container a:hover {
    text-decoration-color: var(--hf-ink);
}
"""

HEADER_HTML = (
    '<div class="header-text">'
    '<h1>Ahead of the Curve</h1>'
    '<p>Explore Hugging Face releases ranked by their forecasted 30-day downloads.</p>'
    '</div>'
)


class PredictionsDashboard:
    """Gradio dashboard that fetches predictions from the inference API and presents them per topic."""

    def __init__(self):
        self.api_url = os.environ["INFERENCE_API_URL"].rstrip("/")
        self.fetch_limit = 500

    def fetch_predictions(self) -> pd.DataFrame:
        url = f"{self.api_url}/predictions"
        resp = httpx.get(url, params={"limit": self.fetch_limit}, timeout=30.0)
        resp.raise_for_status()
        return pd.DataFrame(resp.json())

    def metric_card(self, label: str, value: str, sub: str) -> str:
        return (
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f'</div>'
        )

    def format_timestamp(self, raw) -> str:
        ts = pd.Timestamp(raw)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")

    def empty_state(self, label: str):
        if label == PLACEHOLDER:
            top = self.metric_card("Top predicted model", "—", "Select a topic to view predictions")
            count = self.metric_card("Predicted top quartile", "—", "Select a topic to view predictions")
        else:
            top = self.metric_card("Top predicted model", "—", f"No {label.lower()} predictions yet")
            count = self.metric_card("Predicted top quartile", "0", f"of 0 {label.lower()} predictions")
        return top, count, "", pd.DataFrame(columns=TABLE_HEADERS)

    def on_topic_change(self, label: str):
        if label == PLACEHOLDER:
            return (*self.empty_state(label), gr.update())
        return (*self.render_topic(label), gr.update(choices=list(TOPIC_LABELS.keys())))

    def render_topic(self, label: str):
        try:
            df = self.fetch_predictions()
        except Exception as exc:
            logger.exception("Failed to fetch predictions")
            error = self.metric_card("Error", "—", f"Could not reach inference API: {exc}")
            return error, error, "", pd.DataFrame(columns=TABLE_HEADERS)

        if df.empty or "best_topic" not in df.columns:
            return self.empty_state(label)

        topic = TOPIC_LABELS[label]
        df = df[df["best_topic"] == topic].copy()
        if df.empty:
            return self.empty_state(label)

        df = df.sort_values("downloads_30d_pred", ascending=False).reset_index(drop=True)
        top = df.iloc[0]
        top_quartile_count = int(df["top_quartile_pred"].sum())
        total = len(df)
        predicted_at = self.format_timestamp(df["predicted_at"].iloc[0])

        top_card = self.metric_card(
            "Top predicted model",
            top["model_id"],
            f"{int(round(top['downloads_30d_pred'])):,} predicted downloads (30d)",
        )
        tq_card = self.metric_card(
            "Predicted top quartile",
            f"{top_quartile_count}",
            f"of {total} {label.lower()} predictions",
        )

        display_df = pd.DataFrame({
            "Model": df["model_id"].apply(lambda m: f"[{m}](https://huggingface.co/{m})"),
            "Predicted Downloads (30d)": df["downloads_30d_pred"].round().astype(int),
            "Top Quartile Probability (%)": (df["top_quartile_prob"] * 100).round(1),
            "Top Quartile": df["top_quartile_pred"].map({1: "Yes", 0: "No"}),
        })

        return top_card, tq_card, f"_Last updated: {predicted_at}_", display_df

    def build_ui(self) -> gr.Blocks:
        hf_yellow = gr.themes.Color(
            c50="#FFFDF5", c100="#FFF6BF", c200="#FFEC8A", c300="#FFE259",
            c400="#FFDC36", c500="#FFD21E", c600="#F0B400", c700="#C99500",
            c800="#A37800", c900="#7A5900", c950="#4A3500",
        )
        theme = gr.themes.Soft(
            primary_hue=hf_yellow,
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        )

        with gr.Blocks(theme=theme, css=CSS, title="Ahead of the Curve") as demo:
            gr.HTML(HEADER_HTML)

            with gr.Column(elem_classes=["content"]):
                with gr.Row(elem_classes=["controls-row"]):
                    topic = gr.Dropdown(
                        choices=[PLACEHOLDER] + list(TOPIC_LABELS.keys()),
                        value=PLACEHOLDER,
                        show_label=False,
                        scale=0,
                        container=False,
                        elem_id="topic-dropdown",
                    )
                    refresh = gr.Button(
                        "Refresh",
                        variant="primary",
                        scale=0,
                        elem_id="refresh-btn",
                    )

                with gr.Row(equal_height=True, elem_classes=["cards-row"]):
                    top_card = gr.HTML()
                    tq_card = gr.HTML()

                last_updated = gr.Markdown(elem_classes=["last-updated"])

                table = gr.Dataframe(
                    elem_id="predictions-table",
                    headers=TABLE_HEADERS,
                    datatype=["markdown", "number", "number", "str"],
                    column_widths=["50%", "22%", "18%", "10%"],
                    wrap=True,
                    interactive=False,
                    label="All predictions for selected topic",
                )

            outputs = [top_card, tq_card, last_updated, table, topic]
            topic.change(self.on_topic_change, inputs=topic, outputs=outputs)
            refresh.click(self.on_topic_change, inputs=topic, outputs=outputs)
            demo.load(self.on_topic_change, inputs=topic, outputs=outputs)

        return demo

demo = PredictionsDashboard().build_ui()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
