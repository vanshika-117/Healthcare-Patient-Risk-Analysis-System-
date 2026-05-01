"""
================================================================================
  Healthcare Patient Risk Analysis — Plotly Dash Dashboard
  Interactive ML-powered Dashboard | Resume Project
================================================================================
  Run:  python dashboard.py
  Open: http://127.0.0.1:8050
================================================================================
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from analysis  import generate_patient_data, clean_and_engineer, compute_statistics
from ml_model  import train_models, predict_patient, FEATURES, FEATURE_LABELS, RISK_ORDER

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap data & models at startup
# ─────────────────────────────────────────────────────────────────────────────
print("⚙️   Preparing dataset and training models...")
df      = clean_and_engineer(generate_patient_data(500))
stats   = compute_statistics(df)
results = train_models(df)
rf      = results["rf"]
le      = results["label_encoder"]
print("✅   Ready!\n")

# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#0D1117"
PANEL    = "#161B22"
BORDER   = "#30363D"
TEXT     = "#C9D1D9"
MUTED    = "#8B949E"
ACCENT1  = "#58A6FF"
ACCENT2  = "#3FB950"
ACCENT3  = "#F78166"
ACCENT4  = "#D2A8FF"
YELLOW   = "#E3B341"

RISK_COLORS = {
    "Healthy"       : ACCENT2,
    "Low Risk"      : ACCENT1,
    "Moderate Risk" : YELLOW,
    "High Risk"     : ACCENT3,
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Segoe UI, sans-serif", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=[ACCENT1, ACCENT2, ACCENT3, ACCENT4, YELLOW],
)

def card(children, style=None):
    base = {
        "background"  : PANEL,
        "border"      : f"1px solid {BORDER}",
        "borderRadius": "10px",
        "padding"     : "20px",
        "marginBottom": "16px",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)

def kpi_card(title, value, color=ACCENT1, icon=""):
    return html.Div([
        html.Div(icon, style={"fontSize": "22px", "marginBottom": "6px"}),
        html.Div(str(value), style={
            "fontSize"  : "28px",
            "fontWeight": "700",
            "color"     : color,
        }),
        html.Div(title, style={"fontSize": "12px", "color": MUTED, "marginTop": "4px"}),
    ], style={
        "background"  : PANEL,
        "border"      : f"1.5px solid {color}",
        "borderRadius": "10px",
        "padding"     : "18px",
        "textAlign"   : "center",
        "flex"        : "1",
        "minWidth"    : "130px",
    })

def section_title(text):
    return html.H3(text, style={
        "color"       : TEXT,
        "fontSize"    : "15px",
        "fontWeight"  : "600",
        "marginBottom": "12px",
        "borderBottom": f"1px solid {BORDER}",
        "paddingBottom": "8px",
    })

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute chart figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_risk_donut():
    dist   = stats["risk_distribution"]
    labels = list(dist.keys())
    values = list(dist.values())
    colors = [RISK_COLORS.get(l, MUTED) for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58,
        marker=dict(colors=colors, line=dict(color=BG, width=3)),
        textinfo="percent", textfont=dict(size=12, color=BG),
        hovertemplate="<b>%{label}</b><br>Patients: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Risk Distribution",
                      showlegend=True,
                      legend=dict(orientation="h", y=-0.15, font=dict(size=11)))
    fig.add_annotation(text="Patients", x=0.5, y=0.55, showarrow=False,
                       font=dict(size=11, color=MUTED))
    fig.add_annotation(text=str(len(df)), x=0.5, y=0.42, showarrow=False,
                       font=dict(size=24, color=TEXT, family="Segoe UI"))
    return fig


def fig_age_hist():
    fig = go.Figure(go.Histogram(
        x=df["age"], nbinsx=22, marker_color=ACCENT1,
        marker_line=dict(color=BG, width=0.5),
        hovertemplate="Age %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=df["age"].mean(), line_color=ACCENT3, line_dash="dash",
                  annotation_text=f"Mean: {df['age'].mean():.0f}",
                  annotation_font_color=ACCENT3)
    fig.update_layout(**PLOTLY_LAYOUT, title="Age Distribution",
                      xaxis_title="Age (years)", yaxis_title="Count",
                      bargap=0.05)
    return fig


def fig_bmi_box():
    fig = go.Figure()
    for cat in RISK_ORDER:
        sub = df[df["risk_category"] == cat]["bmi"]
        fig.add_trace(go.Box(
            y=sub, name=cat,
            marker_color=RISK_COLORS[cat],
            line_color=RISK_COLORS[cat],
            fillcolor=RISK_COLORS[cat],
            opacity=0.75,
            boxmean="sd",
            hovertemplate=f"<b>{cat}</b><br>BMI: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(**PLOTLY_LAYOUT, title="BMI by Risk Category",
                      yaxis_title="BMI", showlegend=False)
    return fig


def fig_scatter():
    fig = go.Figure()
    for cat in RISK_ORDER:
        sub = df[df["risk_category"] == cat]
        fig.add_trace(go.Scatter(
            x=sub["cholesterol"], y=sub["glucose"],
            mode="markers", name=cat,
            marker=dict(color=RISK_COLORS[cat], size=6, opacity=0.65,
                        line=dict(width=0, color=BG)),
            hovertemplate=(
                f"<b>{cat}</b><br>"
                "Cholesterol: %{x}<br>Glucose: %{y}<extra></extra>"
            ),
        ))
    fig.add_hline(y=125, line_dash="dash", line_color=ACCENT3,
                  annotation_text="Diabetes threshold (125 mg/dL)",
                  annotation_font_color=ACCENT3)
    fig.update_layout(**PLOTLY_LAYOUT, title="Cholesterol vs Glucose",
                      xaxis_title="Cholesterol (mg/dL)", yaxis_title="Glucose (mg/dL)")
    return fig


def fig_age_risk_bar():
    age_risk = df.groupby("age_group", observed=True)["risk_score"].mean().reset_index()
    fig = go.Figure(go.Bar(
        x=age_risk["age_group"].astype(str),
        y=age_risk["risk_score"],
        marker=dict(
            color=age_risk["risk_score"],
            colorscale=[[0, ACCENT2], [0.5, YELLOW], [1, ACCENT3]],
            showscale=False,
        ),
        text=age_risk["risk_score"].round(2),
        textposition="outside",
        hovertemplate="Age Group: %{x}<br>Avg Risk Score: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Avg Risk Score by Age Group",
                      xaxis_title="Age Group", yaxis_title="Avg Risk Score")
    return fig


def fig_feature_importance():
    imps   = results["feature_importances"]
    labels = [FEATURE_LABELS[f] for f in FEATURES]
    values = [imps[f] for f in FEATURES]
    order  = np.argsort(values)
    fig = go.Figure(go.Bar(
        x=[values[i] for i in order],
        y=[labels[i] for i in order],
        orientation="h",
        marker=dict(
            color=[values[i] for i in order],
            colorscale=[[0, ACCENT1], [1, ACCENT4]],
            showscale=False,
        ),
        text=[f"{values[i]:.3f}" for i in order],
        textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=f"Feature Importance (Random Forest — {results['rf_accuracy']:.1%} Acc)",
                      xaxis_title="Importance Score", yaxis_title="")
    return fig


def fig_confusion_matrix(model="rf"):
    cm     = results[f"{model}_cm"]
    labels = RISK_ORDER
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, PANEL], [0.5, "rgba(88,166,255,0.33)"], [1, ACCENT1]],
        text=cm, texttemplate="%{text}",
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    ))
    title = ("Random Forest" if model == "rf" else "Logistic Regression")
    acc   = results[f"{model}_accuracy"]
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=f"{title} Confusion Matrix (Acc: {acc:.1%})",
                      xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def fig_model_comparison():
    models = ["Random Forest", "Logistic Regression"]
    acc    = [results["rf_accuracy"], results["lr_accuracy"]]
    cv     = [results["rf_cv_accuracy"], results["lr_cv_accuracy"]]
    auc    = [results["rf_auc"], results["lr_auc"]]

    fig = go.Figure()
    for metric, values, color in [
        ("Test Accuracy", acc, ACCENT1),
        ("CV Accuracy",   cv,  ACCENT2),
        ("AUC Score",     auc, ACCENT4),
    ]:
        fig.add_trace(go.Bar(
            name=metric, x=models, y=values,
            marker_color=color,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="Model Performance Comparison",
                      yaxis=dict(range=[0, 1.1], tickformat=".0%"),
                      barmode="group", legend=dict(orientation="h", y=1.1))
    return fig


def fig_correlation():
    cols  = ["age", "bmi", "bp_systolic", "cholesterol", "glucose", "risk_score"]
    short = ["Age", "BMI", "SysBP", "Chol.", "Glucose", "Risk"]
    corr  = df[cols].corr().values
    fig = go.Figure(go.Heatmap(
        z=corr, x=short, y=short,
        colorscale="RdYlGn", zmin=-1, zmax=1,
        text=np.round(corr, 2), texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Feature Correlation Heatmap")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# App Layout
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Healthcare Risk Analysis")
app.config.suppress_callback_exceptions = True

ov = stats["overview"]

TABS_STYLE     = {"borderBottom": f"1px solid {BORDER}", "padding": "6px", "background": BG}
TAB_STYLE      = {"color": MUTED, "background": BG, "border": "none", "borderRadius": "6px 6px 0 0", "padding": "10px 20px"}
TAB_SEL_STYLE  = {"color": TEXT, "background": PANEL, "border": f"1px solid {BORDER}", "borderBottom": f"1px solid {PANEL}", "borderRadius": "6px 6px 0 0", "padding": "10px 20px", "fontWeight": "600"}

app.layout = html.Div([

    # ── Header
    html.Div([
        html.Div([
            html.Span("🏥", style={"fontSize": "28px", "marginRight": "12px"}),
            html.Div([
                html.H1("Healthcare Patient Risk Analysis",
                        style={"margin": "0", "fontSize": "22px", "fontWeight": "700", "color": TEXT}),
                html.P("ML-powered dashboard | Random Forest + Logistic Regression",
                       style={"margin": "0", "color": MUTED, "fontSize": "13px"}),
            ])
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span("● LIVE", style={"color": ACCENT2, "fontSize": "13px", "fontWeight": "600"}),
        ]),
    ], style={
        "display"       : "flex",
        "justifyContent": "space-between",
        "alignItems"    : "center",
        "padding"       : "18px 28px",
        "background"    : PANEL,
        "borderBottom"  : f"1px solid {BORDER}",
    }),

    # ── KPI Row
    html.Div([
        kpi_card("Total Patients",    ov["total_patients"],              ACCENT1, "👤"),
        kpi_card("Average Age",       f"{ov['avg_age']} yrs",            ACCENT2, "🎂"),
        kpi_card("Avg BMI",           ov["avg_bmi"],                     ACCENT4, "⚖️"),
        kpi_card("Hypertension",      f"{ov['hypertension_pct']}%",      ACCENT3, "❤️"),
        kpi_card("Diabetes Risk",     f"{ov['diabetes_risk_pct']}%",     YELLOW,  "🩺"),
        kpi_card("RF Accuracy",       f"{results['rf_accuracy']:.1%}",   ACCENT2, "🤖"),
        kpi_card("RF AUC Score",      f"{results['rf_auc']:.3f}",        ACCENT1, "📈"),
    ], style={
        "display"       : "flex",
        "gap"           : "12px",
        "padding"       : "20px 28px 0",
        "flexWrap"      : "wrap",
    }),

    # ── Tabs
    html.Div([
        dcc.Tabs(id="tabs", value="overview", style=TABS_STYLE, children=[
            dcc.Tab(label="📊 Overview",           value="overview",   style=TAB_STYLE, selected_style=TAB_SEL_STYLE),
            dcc.Tab(label="🤖 ML & Models",        value="ml",         style=TAB_STYLE, selected_style=TAB_SEL_STYLE),
            dcc.Tab(label="🔮 Risk Predictor",     value="predictor",  style=TAB_STYLE, selected_style=TAB_SEL_STYLE),
            dcc.Tab(label="🗃️  Patient Table",     value="table",      style=TAB_STYLE, selected_style=TAB_SEL_STYLE),
        ]),
        html.Div(id="tab-content"),
    ], style={"padding": "20px 28px"}),

], style={"background": BG, "minHeight": "100vh", "fontFamily": "Segoe UI, sans-serif"})


# ─────────────────────────────────────────────────────────────────────────────
# Tab content callback
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):

    # ── OVERVIEW TAB
    if tab == "overview":
        return html.Div([
            html.Div([
                card([dcc.Graph(figure=fig_risk_donut(),   config={"displayModeBar": False})], style={"flex":"1"}),
                card([dcc.Graph(figure=fig_age_hist(),     config={"displayModeBar": False})], style={"flex":"1"}),
                card([dcc.Graph(figure=fig_bmi_box(),      config={"displayModeBar": False})], style={"flex":"1"}),
            ], style={"display":"flex", "gap":"16px"}),
            html.Div([
                card([dcc.Graph(figure=fig_scatter(),      config={"displayModeBar": False})], style={"flex":"1"}),
                card([dcc.Graph(figure=fig_age_risk_bar(), config={"displayModeBar": False})], style={"flex":"1"}),
                card([dcc.Graph(figure=fig_correlation(),  config={"displayModeBar": False})], style={"flex":"1"}),
            ], style={"display":"flex", "gap":"16px"}),
        ])

    # ── ML TAB
    elif tab == "ml":
        return html.Div([
            html.Div([
                card([dcc.Graph(figure=fig_model_comparison(),      config={"displayModeBar": False})], style={"flex":"1"}),
                card([dcc.Graph(figure=fig_feature_importance(),    config={"displayModeBar": False})], style={"flex":"1"}),
            ], style={"display":"flex", "gap":"16px"}),
            html.Div([
                card([
                    section_title("Random Forest Confusion Matrix"),
                    dcc.Graph(figure=fig_confusion_matrix("rf"), config={"displayModeBar": False}),
                ], style={"flex":"1"}),
                card([
                    section_title("Logistic Regression Confusion Matrix"),
                    dcc.Graph(figure=fig_confusion_matrix("lr"), config={"displayModeBar": False}),
                ], style={"flex":"1"}),
                card([
                    section_title("Model Metrics Summary"),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Metric",     style={"color": MUTED, "padding":"8px", "textAlign":"left"}),
                            html.Th("Random Forest", style={"color": ACCENT1, "padding":"8px", "textAlign":"center"}),
                            html.Th("Log. Reg.",     style={"color": ACCENT4, "padding":"8px", "textAlign":"center"}),
                        ])),
                        html.Tbody([
                            html.Tr([html.Td("Test Accuracy", style={"padding":"6px","color":TEXT}),
                                     html.Td(f"{results['rf_accuracy']:.1%}", style={"color":ACCENT1,"textAlign":"center","padding":"6px","fontWeight":"600"}),
                                     html.Td(f"{results['lr_accuracy']:.1%}", style={"color":ACCENT4,"textAlign":"center","padding":"6px","fontWeight":"600"})]),
                            html.Tr([html.Td("CV Accuracy (5-fold)", style={"padding":"6px","color":TEXT}),
                                     html.Td(f"{results['rf_cv_accuracy']:.1%}", style={"color":ACCENT1,"textAlign":"center","padding":"6px"}),
                                     html.Td(f"{results['lr_cv_accuracy']:.1%}", style={"color":ACCENT4,"textAlign":"center","padding":"6px"})]),
                            html.Tr([html.Td("AUC Score (macro OvR)", style={"padding":"6px","color":TEXT}),
                                     html.Td(f"{results['rf_auc']:.3f}", style={"color":ACCENT1,"textAlign":"center","padding":"6px"}),
                                     html.Td(f"{results['lr_auc']:.3f}", style={"color":ACCENT4,"textAlign":"center","padding":"6px"})]),
                            html.Tr([html.Td("Training samples", style={"padding":"6px","color":TEXT}),
                                     html.Td(str(results["n_train"]), style={"color":MUTED,"textAlign":"center","padding":"6px"}),
                                     html.Td(str(results["n_train"]), style={"color":MUTED,"textAlign":"center","padding":"6px"})]),
                            html.Tr([html.Td("Test samples", style={"padding":"6px","color":TEXT}),
                                     html.Td(str(results["n_test"]), style={"color":MUTED,"textAlign":"center","padding":"6px"}),
                                     html.Td(str(results["n_test"]), style={"color":MUTED,"textAlign":"center","padding":"6px"})]),
                        ]),
                    ], style={"width":"100%", "borderCollapse":"collapse"}),
                ], style={"flex":"1"}),
            ], style={"display":"flex", "gap":"16px"}),
        ])

    # ── PREDICTOR TAB
    elif tab == "predictor":
        def slider(label, id, min, max, step, value, marks=None):
            return html.Div([
                html.Label(f"{label}: ", style={"color": MUTED, "fontSize": "13px"}),
                html.Span(id=f"{id}-display", style={"color": ACCENT1, "fontWeight": "600", "fontSize": "13px"}),
                dcc.Slider(id=id, min=min, max=max, step=step, value=value,
                           marks=marks or {min: str(min), max: str(max)},
                           tooltip={"placement": "bottom", "always_visible": False},
                           updatemode="drag"),
            ], style={"marginBottom": "18px"})

        return html.Div([
            html.Div([
                card([
                    section_title("Patient Parameters"),
                    slider("Age",              "s-age",      18,  90,  1,  45),
                    slider("BMI",              "s-bmi",      15,  50,  0.5, 27),
                    slider("Systolic BP",      "s-bps",      90, 200,  1, 120),
                    slider("Diastolic BP",     "s-bpd",      60, 120,  1,  80),
                    slider("Cholesterol",      "s-chol",    120, 320,  1, 180),
                    slider("Glucose",          "s-gluc",     70, 300,  1,  90),
                    slider("Exercise Days/wk", "s-exdays",   0,   7,  1,   3,
                           {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7"}),
                    html.Div([
                        html.Label("Smoker:", style={"color": MUTED, "fontSize": "13px", "marginRight": "12px"}),
                        dcc.RadioItems(
                            id="s-smoker",
                            options=[{"label": " No", "value": 0}, {"label": " Yes", "value": 1}],
                            value=0, inline=True,
                            labelStyle={"marginRight": "20px", "color": TEXT, "fontSize": "13px"},
                        ),
                    ], style={"marginBottom": "18px"}),
                    html.Button("🔮  Predict Risk", id="btn-predict",
                                style={
                                    "width"       : "100%",
                                    "padding"     : "12px",
                                    "background"  : ACCENT1,
                                    "color"       : BG,
                                    "border"      : "none",
                                    "borderRadius": "8px",
                                    "fontWeight"  : "700",
                                    "fontSize"    : "15px",
                                    "cursor"      : "pointer",
                                }),
                ], style={"flex": "1", "minWidth": "300px"}),

                # Results panel
                html.Div(id="prediction-result", style={"flex": "2"}),
            ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),
        ])

    # ── TABLE TAB
    elif tab == "table":
        cols_show = ["patient_id","age","gender","bmi","bp_systolic",
                     "cholesterol","glucose","risk_score","risk_category"]
        return card([
            section_title(f"Patient Records — {len(df)} total"),
            html.Div([
                html.Label("Filter by Risk Category:", style={"color": MUTED, "fontSize": "13px", "marginRight": "8px"}),
                dcc.Dropdown(
                    id="tbl-filter",
                    options=[{"label": "All", "value": "All"}] +
                            [{"label": c, "value": c} for c in RISK_ORDER],
                    value="All", clearable=False,
                    style={"width": "220px", "background": PANEL, "color": BG},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "14px"}),
            dash_table.DataTable(
                id="patient-table",
                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols_show],
                data=df[cols_show].to_dict("records"),
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": BG, "color": MUTED,
                    "fontWeight": "600", "border": f"1px solid {BORDER}",
                    "fontSize": "12px",
                },
                style_cell={
                    "backgroundColor": PANEL, "color": TEXT,
                    "border": f"1px solid {BORDER}", "fontSize": "13px",
                    "padding": "8px 12px", "textAlign": "center",
                },
                style_data_conditional=[
                    {"if": {"filter_query": '{risk_category} = "High Risk"'},
                     "color": ACCENT3, "fontWeight": "600"},
                    {"if": {"filter_query": '{risk_category} = "Moderate Risk"'},
                     "color": YELLOW},
                    {"if": {"filter_query": '{risk_category} = "Low Risk"'},
                     "color": ACCENT1},
                    {"if": {"filter_query": '{risk_category} = "Healthy"'},
                     "color": ACCENT2},
                ],
            ),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Predictor callback
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("prediction-result", "children"),
    Input("btn-predict", "n_clicks"),
    State("s-age",    "value"), State("s-bmi",    "value"),
    State("s-bps",    "value"), State("s-bpd",    "value"),
    State("s-chol",   "value"), State("s-gluc",   "value"),
    State("s-smoker", "value"), State("s-exdays", "value"),
    prevent_initial_call=True,
)
def run_prediction(n, age, bmi, bps, bpd, chol, gluc, smoker, exdays):
    patient = dict(age=age, bmi=bmi, bp_systolic=bps, bp_diastolic=bpd,
                   cholesterol=chol, glucose=gluc, smoker=smoker, exercise_days=exdays)
    res = predict_patient(patient, rf, le)
    cat   = res["category"]
    color = RISK_COLORS.get(cat, ACCENT1)
    probs = res["probabilities"]

    prob_fig = go.Figure(go.Bar(
        x=list(probs.values()), y=list(probs.keys()),
        orientation="h",
        marker=dict(color=[RISK_COLORS.get(k, MUTED) for k in probs.keys()]),
        text=[f"{v:.1f}%" for v in probs.values()],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    prob_layout = {**PLOTLY_LAYOUT, "margin": dict(l=120, r=40, t=40, b=20)}
    prob_fig.update_layout(
        **prob_layout, title="Prediction Probabilities",
        xaxis=dict(range=[0, 110], ticksuffix="%"),
        height=220,
    )

    recs = res["recommendations"]
    return html.Div([
        card([
            html.Div([
                html.Div("Prediction Result", style={"color": MUTED, "fontSize": "13px", "marginBottom": "6px"}),
                html.Div(cat, style={"fontSize": "34px", "fontWeight": "800", "color": color}),
                html.Div(f"Confidence: {res['confidence']}%",
                         style={"color": MUTED, "fontSize": "13px", "marginTop": "4px"}),
            ], style={"textAlign": "center", "padding": "10px 0"}),
        ]),
        card([dcc.Graph(figure=prob_fig, config={"displayModeBar": False})]),
        card([
            section_title("💡 Personalised Recommendations"),
            html.Ul([
                html.Li(r, style={"color": TEXT, "marginBottom": "8px", "fontSize": "13px"})
                for r in recs
            ]) if recs else html.P("No risk factors identified. Keep up the healthy habits! ✅",
                                   style={"color": ACCENT2, "fontSize": "13px"}),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Table filter callback
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("patient-table", "data"),
    Input("tbl-filter", "value"),
    prevent_initial_call=False,
)
def filter_table(category):
    cols_show = ["patient_id","age","gender","bmi","bp_systolic",
                 "cholesterol","glucose","risk_score","risk_category"]
    if category == "All":
        return df[cols_show].to_dict("records")
    return df[df["risk_category"] == category][cols_show].to_dict("records")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀  Dashboard running at http://127.0.0.1:8050\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
