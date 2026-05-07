"""
NYC Taxi Fare Prediction - Gradio Interface
=============================================
Modern, premium UI with Navy + Orange theme.
"""

import gradio as gr
import numpy as np
import joblib
import json
import os

# Load Model Artifacts
model = joblib.load("ann_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")

metrics = {}
if os.path.exists("metrics.json"):
    with open("metrics.json") as f:
        metrics = json.load(f)


def predict_fare(
    vendor_id, passenger_count, trip_distance, rate_code,
    store_and_fwd, pu_location, do_location, payment_type,
    extra, mta_tax, tip_amount, tolls_amount,
    improvement_surcharge, congestion_surcharge,
    pickup_hour, pickup_day, pickup_month,
    is_weekend, is_rush_hour, is_night,
    trip_duration_min, avg_speed_mph
):
    features = np.array([[
        vendor_id, passenger_count, trip_distance, rate_code,
        1 if store_and_fwd == "Yes" else 0,
        pu_location, do_location, payment_type,
        extra, mta_tax, tip_amount, tolls_amount,
        improvement_surcharge, congestion_surcharge,
        pickup_hour, pickup_day, pickup_month,
        1 if is_weekend else 0,
        1 if is_rush_hour else 0,
        1 if is_night else 0,
        trip_duration_min, avg_speed_mph
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prediction = max(2.5, prediction)

    total_est = prediction + extra + mta_tax + tip_amount + tolls_amount + improvement_surcharge + congestion_surcharge

    return (
        f"<div class='result-card'>"
        f"<div class='result-label'>PREDICTED BASE FARE</div>"
        f"<div class='result-amount'>${prediction:.2f}</div>"
        f"<div class='result-divider'></div>"
        f"<div class='result-label'>ESTIMATED TOTAL</div>"
        f"<div class='result-total'>${total_est:.2f}</div>"
        f"<div class='result-divider'></div>"
        f"<table class='breakdown-table'>"
        f"<tr><td>Base Fare</td><td>${prediction:.2f}</td></tr>"
        f"<tr><td>Extra</td><td>${extra:.2f}</td></tr>"
        f"<tr><td>MTA Tax</td><td>${mta_tax:.2f}</td></tr>"
        f"<tr><td>Tip</td><td>${tip_amount:.2f}</td></tr>"
        f"<tr><td>Tolls</td><td>${tolls_amount:.2f}</td></tr>"
        f"<tr><td>Improvement Surcharge</td><td>${improvement_surcharge:.2f}</td></tr>"
        f"<tr><td>Congestion Surcharge</td><td>${congestion_surcharge:.2f}</td></tr>"
        f"<tr class='total-row'><td>Total</td><td>${total_est:.2f}</td></tr>"
        f"</table>"
        f"</div>"
    )


# ── Custom CSS ───────────────────────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Global ── */
* { font-family: 'Inter', sans-serif !important; }

.gradio-container {
    background: #0a0f1e !important;
    max-width: 1200px !important;
    margin: auto !important;
}

/* ── Header ── */
#header-block {
    background: linear-gradient(135deg, #0d1529 0%, #131d3b 100%) !important;
    border: 1px solid rgba(255, 140, 50, 0.15) !important;
    border-radius: 20px !important;
    padding: 32px !important;
    text-align: center;
    margin-bottom: 8px !important;
    position: relative;
    overflow: hidden;
}
#header-block::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(255, 120, 30, 0.06) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(255, 160, 60, 0.04) 0%, transparent 50%);
    pointer-events: none;
}
#header-block h1 {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    margin: 0 0 4px 0 !important;
}
#header-block p {
    color: rgba(255, 255, 255, 0.5) !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    margin: 0 !important;
}
.orange-text { color: #ff8c32 !important; }

/* ── Tabs ── */
.tabs { border: none !important; }
.tab-nav {
    background: transparent !important;
    border: none !important;
    gap: 4px !important;
    padding: 0 !important;
    margin-bottom: 12px !important;
}
.tab-nav button {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    color: rgba(255, 255, 255, 0.4) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.tab-nav button:hover {
    background: rgba(255, 140, 50, 0.08) !important;
    color: rgba(255, 255, 255, 0.7) !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #ff8c32, #ff6a10) !important;
    color: #ffffff !important;
    border-color: transparent !important;
    box-shadow: 0 4px 20px rgba(255, 140, 50, 0.3) !important;
}

/* ── Cards / Groups ── */
.gr-group, .gr-box, .gr-panel, .block {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
}

/* ── Labels ── */
label, .gr-input-label, span.text-lg {
    color: rgba(255, 255, 255, 0.7) !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
}

/* ── Inputs ── */
input[type="number"],
textarea,
.gr-input,
.gr-text-input {
    background: #0d1529 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    font-size: 0.95rem !important;
    transition: all 0.25s ease !important;
}
input[type="number"]:focus,
textarea:focus {
    border-color: #ff8c32 !important;
    box-shadow: 0 0 0 3px rgba(255, 140, 50, 0.12) !important;
    outline: none !important;
}

/* ── Dropdowns ── */
.gr-dropdown, select {
    background: #0d1529 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
}

/* ── Sliders ── */
input[type="range"] { accent-color: #ff8c32 !important; }
.range-slider .thumb { background: #ff8c32 !important; }
.range-slider .track-fill { background: #ff8c32 !important; }
span.number-input { background: transparent !important; }

/* ── Checkboxes ── */
input[type="checkbox"] { accent-color: #ff8c32 !important; }
.gr-check-radio { border-color: rgba(255, 140, 50, 0.4) !important; }
.gr-check-radio.checked { background: #ff8c32 !important; border-color: #ff8c32 !important; }

/* ── Accordion ── */
.accordion {
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    background: rgba(255, 255, 255, 0.015) !important;
}
.accordion .label-wrap {
    color: rgba(255, 255, 255, 0.6) !important;
    font-weight: 600 !important;
}

/* ── Primary Button ── */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #ff8c32, #ff6a10) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 40px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(255, 140, 50, 0.25) !important;
    cursor: pointer !important;
}
.gr-button-primary:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(255, 140, 50, 0.4) !important;
    background: linear-gradient(135deg, #ffa04a, #ff8c32) !important;
}
.gr-button-primary:active, button.primary:active {
    transform: translateY(0px) !important;
}

/* ── Output / Result Card ── */
.result-card {
    background: linear-gradient(135deg, #0d1529 0%, #111b35 100%);
    border: 1px solid rgba(255, 140, 50, 0.2);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
}
.result-label {
    color: rgba(255, 255, 255, 0.4);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.result-amount {
    color: #ff8c32;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 16px;
}
.result-total {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 20px;
}
.result-divider {
    height: 1px;
    background: rgba(255, 255, 255, 0.06);
    margin: 16px 0;
}
.breakdown-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 4px;
}
.breakdown-table td {
    padding: 8px 12px;
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.5);
}
.breakdown-table td:last-child {
    text-align: right;
    color: rgba(255, 255, 255, 0.7);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}
.breakdown-table tr:hover td {
    background: rgba(255, 140, 50, 0.04);
    border-radius: 8px;
}
.breakdown-table .total-row td {
    color: #ff8c32 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border-top: 1px solid rgba(255, 140, 50, 0.2);
    padding-top: 12px;
}

/* ── Markdown ── */
.gr-markdown, .markdown-text, .prose {
    color: rgba(255, 255, 255, 0.6) !important;
}
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}
.gr-markdown strong { color: #ff8c32 !important; }
.gr-markdown table { border-collapse: collapse; width: 100%; }
.gr-markdown th {
    background: rgba(255, 140, 50, 0.1);
    color: #ff8c32;
    font-weight: 600;
    padding: 10px 16px;
    text-align: left;
    border-bottom: 2px solid rgba(255, 140, 50, 0.2);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.gr-markdown td {
    padding: 10px 16px;
    color: rgba(255, 255, 255, 0.6);
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 0.9rem;
}
.gr-markdown tr:hover td { background: rgba(255, 140, 50, 0.03); }

/* ── Section Labels ── */
.section-label {
    color: #ff8c32 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}

/* ── Info text ── */
.gr-info, .info { color: rgba(255, 255, 255, 0.3) !important; font-size: 0.75rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e2a4a; border-radius: 10px; }

/* ── Footer ── */
footer { display: none !important; }

/* ── Image styling ── */
.gr-image { border-radius: 16px !important; overflow: hidden !important; }
"""


def get_metrics_md():
    if not metrics:
        return "Metrics not available."
    m = metrics.get("metrics", {})
    return (
        f"## Model Performance\n\n"
        f"**Algorithm:** {metrics.get('model', 'ANN')}\n\n"
        f"**Architecture:** {metrics.get('architecture', 'N/A')} neurons\n\n"
        f"**Training Samples:** {metrics.get('training_samples', 'N/A'):,} | "
        f"**Test Samples:** {metrics.get('test_samples', 'N/A'):,}\n\n"
        f"| Metric | Train | Test |\n"
        f"|--------|-------|------|\n"
        f"| MAE ($) | {m.get('train_mae', '-')} | {m.get('test_mae', '-')} |\n"
        f"| RMSE ($) | {m.get('train_rmse', '-')} | {m.get('test_rmse', '-')} |\n"
        f"| R-Squared | {m.get('train_r2', '-')} | {m.get('test_r2', '-')} |\n"
        f"| MAPE (%) | {m.get('train_mape', '-')} | {m.get('test_mape', '-')} |\n\n"
        f"**Training Time:** {metrics.get('training_time_seconds', '-')}s | "
        f"**Iterations:** {metrics.get('iterations', '-')}\n"
    )


# ── Build Interface ──────────────────────────────────────────────────────────────
with gr.Blocks(css=custom_css, title="NYC Taxi Fare Predictor", theme=gr.themes.Base()) as demo:

    gr.HTML(
        """
        <div id="header-block">
            <h1>NYC Taxi Fare <span class="orange-text">Predictor</span></h1>
            <p>Artificial Neural Network trained on 6.4M NYC Yellow Taxi trip records</p>
        </div>
        """,
        elem_id="header-block-wrapper"
    )

    with gr.Tabs():

        # ── Tab 1: Predict ──
        with gr.Tab("Predict"):
            with gr.Row():
                with gr.Column(scale=3):

                    gr.Markdown("<span class='section-label'>Trip Details</span>")
                    with gr.Row():
                        vendor_id = gr.Dropdown([1, 2], value=1, label="Vendor", info="1 = CMT, 2 = VFI")
                        passenger_count = gr.Slider(1, 6, value=1, step=1, label="Passengers")
                    with gr.Row():
                        trip_distance = gr.Number(value=3.5, label="Trip Distance (miles)", minimum=0)
                        trip_duration = gr.Number(value=15.0, label="Trip Duration (min)", minimum=0)
                    with gr.Row():
                        avg_speed = gr.Number(value=14.0, label="Avg Speed (mph)", minimum=0)
                        rate_code = gr.Dropdown(
                            [1, 2, 3, 4, 5, 6], value=1, label="Rate Code",
                            info="1=Standard 2=JFK 3=Newark 4=Nassau 5=Negotiated 6=Group"
                        )

                    gr.Markdown("<span class='section-label'>Locations</span>")
                    with gr.Row():
                        pu_location = gr.Number(value=161, label="Pickup Location ID", minimum=1, maximum=265)
                        do_location = gr.Number(value=237, label="Dropoff Location ID", minimum=1, maximum=265)

                    gr.Markdown("<span class='section-label'>Time</span>")
                    with gr.Row():
                        pickup_hour = gr.Slider(0, 23, value=14, step=1, label="Hour")
                        pickup_day = gr.Slider(0, 6, value=2, step=1, label="Day of Week (0=Mon)")
                        pickup_month = gr.Slider(1, 12, value=1, step=1, label="Month")
                    with gr.Row():
                        is_weekend = gr.Checkbox(value=False, label="Weekend")
                        is_rush = gr.Checkbox(value=False, label="Rush Hour")
                        is_night = gr.Checkbox(value=False, label="Night")

                    with gr.Accordion("Payment and Surcharges", open=False):
                        with gr.Row():
                            payment_type = gr.Dropdown(
                                [1, 2, 3, 4], value=1, label="Payment Type",
                                info="1=Credit 2=Cash 3=No Charge 4=Dispute"
                            )
                            store_fwd = gr.Dropdown(["No", "Yes"], value="No", label="Store and Forward")
                        with gr.Row():
                            extra = gr.Number(value=0.5, label="Extra ($)")
                            mta_tax = gr.Number(value=0.5, label="MTA Tax ($)")
                        with gr.Row():
                            tip = gr.Number(value=2.0, label="Tip ($)")
                            tolls = gr.Number(value=0.0, label="Tolls ($)")
                        with gr.Row():
                            imp_surcharge = gr.Number(value=0.3, label="Improvement Surcharge ($)")
                            cong_surcharge = gr.Number(value=2.5, label="Congestion Surcharge ($)")

                    predict_btn = gr.Button("Predict Fare", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("<span class='section-label'>Prediction Result</span>")
                    output = gr.HTML(
                        "<div class='result-card'>"
                        "<div class='result-label'>WAITING FOR INPUT</div>"
                        "<div class='result-amount' style='color: rgba(255,255,255,0.15);'>$0.00</div>"
                        "</div>"
                    )

            predict_btn.click(
                fn=predict_fare,
                inputs=[
                    vendor_id, passenger_count, trip_distance, rate_code,
                    store_fwd, pu_location, do_location, payment_type,
                    extra, mta_tax, tip, tolls,
                    imp_surcharge, cong_surcharge,
                    pickup_hour, pickup_day, pickup_month,
                    is_weekend, is_rush, is_night,
                    trip_duration, avg_speed
                ],
                outputs=output
            )

        # ── Tab 2: Model Info ──
        with gr.Tab("Performance"):
            gr.Markdown(get_metrics_md())
            if os.path.exists("evaluation_plots.png"):
                gr.Image("evaluation_plots.png", label="Evaluation Plots", show_label=False)

        # ── Tab 3: About ──
        with gr.Tab("About"):
            gr.Markdown(
                """
                ## About This Project

                **NYC Taxi Fare Prediction** using an Artificial Neural Network
                trained on the NYC Yellow Taxi Trip Records dataset (2020).

                ### Model Details
                - **Algorithm:** MLPRegressor (scikit-learn)
                - **Architecture:** 128 > 64 > 32 neurons
                - **Activation:** ReLU
                - **Optimizer:** Adam with adaptive learning rate and early stopping
                - **Dataset:** ~6.4 million trip records, sampled to ~960K for training

                ### Features Used (22 total)
                Trip distance, duration, average speed, pickup/dropoff location IDs,
                vendor, passenger count, rate code, payment type, time features
                (hour, day of week, month), binary flags (weekend, rush hour, night),
                and surcharge amounts.
                """
            )

if __name__ == "__main__":
    demo.launch()
