"""
PredictMaint – Maintenance prédictive des machines industrielles
Projet PFA – Yousra Erraki (EMSI)

Lancer :  streamlit run app.py
Données : placer DataPiece.csv (séparateur ;) à côté de app.py, ou le charger dans la barre latérale.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="PredictMaint", page_icon="⚙️", layout="wide",
                   initial_sidebar_state="expanded")

# ─────────────────────────── Style ───────────────────────────
ACCENT, OK, DANGER, INK, MUTED = "#F97316", "#10B981", "#EF4444", "#0F172A", "#64748B"
PALETTE = [ACCENT, "#0EA5E9", "#8B5CF6", OK, "#EAB308", "#EC4899"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; max-width: 1250px; }}

[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: #E2E8F0 !important; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{ background: #1E293B; border: 1px dashed #475569; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{ background: {ACCENT}; border: 0; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {{ color: white !important; }}

.hero {{
  background: linear-gradient(120deg, {INK} 0%, #1E293B 55%, #7C2D12 100%);
  border-radius: 20px; padding: 30px 34px; color: white; margin-bottom: 22px;
  position: relative; overflow: hidden;
}}
.hero:after {{ content: "⚙"; position: absolute; right: 26px; top: -18px; font-size: 170px; opacity: .08; }}
.hero .tag {{ display:inline-block; background: {ACCENT}; color: white; font-weight: 700; font-size: 12px;
  letter-spacing: .08em; padding: 4px 12px; border-radius: 999px; text-transform: uppercase; }}
.hero h1 {{ font-size: 34px; font-weight: 800; margin: 12px 0 6px; color: white; }}
.hero p {{ color: #CBD5E1; font-size: 15.5px; margin: 0; max-width: 760px; }}

.kpi {{ background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 18px 20px;
  box-shadow: 0 1px 3px rgba(15,23,42,.06); height: 100%; }}
.kpi .label {{ color: {MUTED}; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}
.kpi .value {{ color: {INK}; font-size: 30px; font-weight: 800; margin-top: 4px; }}
.kpi .sub {{ color: {MUTED}; font-size: 13px; }}

.card {{ background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 20px 22px;
  box-shadow: 0 1px 3px rgba(15,23,42,.06); }}
.section {{ font-size: 13px; font-weight: 700; color: {ACCENT}; text-transform: uppercase;
  letter-spacing: .08em; margin: 6px 0 2px; }}

.verdict {{ border-radius: 16px; padding: 20px 22px; color: white; }}
.verdict h3 {{ margin: 0 0 6px; font-size: 22px; font-weight: 800; color: white; }}
.verdict p {{ margin: 0; opacity: .92; }}
.chip {{ display:inline-block; background: #F1F5F9; color: {INK}; border-radius: 999px; padding: 5px 12px;
  margin: 4px 6px 0 0; font-size: 13px; font-weight: 600; }}
.chip.up {{ background: #FEE2E2; color: #991B1B; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
.stTabs [data-baseweb="tab"] p {{ font-weight: 600; font-size: 15px; }}
.stTabs [aria-selected="true"] p {{ color: {ACCENT} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {ACCENT} !important; height: 3px; }}
.stButton > button {{ background: {ACCENT}; color: white; border: 0; border-radius: 12px;
  font-weight: 700; padding: 12px; font-size: 16px; }}
.stButton > button:hover {{ background: #EA580C; color: white; }}
</style>
""", unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, font=dict(family="Inter", color=INK),
                      margin=dict(l=10, r=10, t=50, b=10), title_font=dict(size=16),
                      colorway=PALETTE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ─────────────────────────── Données & modèle ───────────────────────────
TARGET = "breakdown_flag"
CAT_COLS = ["machine_type", "plant_code", "part_family", "criticality", "uom"]
NUM_COLS = ["unit_cost_inr", "qty_issued", "temp_bearing_degC", "temp_motor_degC",
            "vibration_h_mms", "vibration_v_mms", "oil_pressure_bar", "load_pct",
            "shaft_rpm", "power_consumption_kw"]
SENSORS = NUM_COLS[2:]
LABELS = {
    "machine_type": "Type de machine", "plant_code": "Site", "part_family": "Famille de pièce",
    "criticality": "Criticité", "uom": "Unité", "unit_cost_inr": "Coût unitaire (INR)",
    "qty_issued": "Quantité sortie", "temp_bearing_degC": "Temp. roulement (°C)",
    "temp_motor_degC": "Temp. moteur (°C)", "vibration_h_mms": "Vibration horiz. (mm/s)",
    "vibration_v_mms": "Vibration vert. (mm/s)", "oil_pressure_bar": "Pression d'huile (bar)",
    "load_pct": "Charge (%)", "shaft_rpm": "Vitesse arbre (tr/min)",
    "power_consumption_kw": "Consommation (kW)", "mois": "Mois", "saison": "Saison",
}


@st.cache_data(show_spinner=False)
def load_data(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), sep=";")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], dayfirst=True, errors="coerce")
    df = df.dropna()
    df["mois"] = df["transaction_date"].dt.month
    df["saison"] = df["mois"].map(lambda m: 1 if m in (12, 1, 2) else 2 if m in (3, 4, 5)
                                  else 3 if m in (6, 7, 8) else 4)
    return df


def encode(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    X = df[CAT_COLS + NUM_COLS + ["mois", "saison"]].copy()
    for col in CAT_COLS:
        X[col] = X[col].astype(str).map(mappings[col]).fillna(-1).astype(int)
    return X


@st.cache_resource(show_spinner=False)
def train(df: pd.DataFrame):
    pannes, normales = df[df[TARGET] == 1], df[df[TARGET] == 0]
    n = min(len(normales), len(pannes) * 3)  # 3 non-pannes pour 1 panne, comme dans le notebook
    data = pd.concat([normales.sample(n=n, random_state=42), pannes]).sample(frac=1, random_state=42)
    mappings = {c: {v: i for i, v in enumerate(sorted(data[c].astype(str).unique()))} for c in CAT_COLS}
    X, y = encode(data, mappings), data[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", min_samples_leaf=5,
                                   random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    return model, mappings, X_te, y_te, model.predict_proba(X_te)[:, 1]


# ─────────────────────────── Barre latérale ───────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ PredictMaint")
    st.caption("Maintenance prédictive par IA")
    st.markdown("---")
    fichier = st.file_uploader("Jeu de données (CSV, séparateur ;)", type="csv")
    seuil = st.slider("Seuil d'alerte", 0.10, 0.90, 0.40, 0.05,
                      help="Plus le seuil est bas, moins on rate de pannes, mais plus il y a de fausses alertes.")
    st.markdown("---")
    st.caption("Projet PFA · Yousra Erraki\nEMSI · IA & Data Science")

st.markdown("""
<div class="hero">
  <span class="tag">Maintenance prédictive · IA</span>
  <h1>Anticiper les pannes avant qu'elles n'arrivent</h1>
  <p>Le modèle analyse les capteurs des machines (températures, vibrations, pression d'huile, charge…)
  et estime le risque de panne pour planifier la maintenance au bon moment.</p>
</div>
""", unsafe_allow_html=True)

local_csv = Path(__file__).with_name("DataPiece.csv")
if fichier is not None:
    raw = fichier.getvalue()
elif local_csv.exists():
    raw = local_csv.read_bytes()
else:
    st.info("👈 Chargez **DataPiece.csv** dans la barre latérale, ou placez-le dans le même dossier que app.py.")
    st.stop()

with st.spinner("Chargement des données et entraînement du modèle…"):
    df = load_data(raw)
    model, mappings, X_te, y_te, proba_te = train(df)

y_pred_te = (proba_te >= seuil).astype(int)
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Enregistrements", f"{len(df):,}".replace(",", " "), "historique analysé")
kpi(k2, "Machines suivies", df["machine_type"].nunique(), f"sur {df['plant_code'].nunique()} sites")
kpi(k3, "Taux de panne", f"{df[TARGET].mean():.1%}", "dans l'historique")
kpi(k4, "Pannes détectées", f"{recall_score(y_te, y_pred_te, zero_division=0):.0%}",
    f"au seuil de {seuil:.0%}")
st.write("")

tab_pred, tab_data, tab_model = st.tabs(["🔮  Diagnostic", "📊  Exploration", "🧠  Performance du modèle"])

# ─────────────────────────── Diagnostic ───────────────────────────
with tab_pred:
    med = df[NUM_COLS].median()
    q90 = df[SENSORS].quantile(0.90)
    left, right = st.columns([1.35, 1], gap="large")
    entree = {}
    with left:
        st.markdown('<div class="section">Machine</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        entree["machine_type"] = a.selectbox(LABELS["machine_type"], sorted(mappings["machine_type"]))
        entree["plant_code"] = b.selectbox(LABELS["plant_code"], sorted(mappings["plant_code"]))
        a, b, c = st.columns(3)
        entree["part_family"] = a.selectbox(LABELS["part_family"], sorted(mappings["part_family"]))
        entree["criticality"] = b.selectbox(LABELS["criticality"], sorted(mappings["criticality"]))
        entree["uom"] = c.selectbox(LABELS["uom"], sorted(mappings["uom"]))

        st.markdown('<div class="section">Capteurs</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, col in enumerate(SENSORS):
            entree[col] = cols[i % 2].number_input(LABELS[col], value=float(round(med[col], 2)))

        st.markdown('<div class="section">Pièce & date</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        entree["unit_cost_inr"] = a.number_input(LABELS["unit_cost_inr"], value=float(med["unit_cost_inr"]))
        entree["qty_issued"] = b.number_input(LABELS["qty_issued"], value=float(med["qty_issued"]), step=1.0)
        date = c.date_input("Date")
        m = date.month
        entree["mois"] = m
        entree["saison"] = 1 if m in (12, 1, 2) else 2 if m in (3, 4, 5) else 3 if m in (6, 7, 8) else 4
        lancer = st.button("Lancer le diagnostic", width="stretch")

    with right:
        if not lancer:
            st.markdown('<div class="card" style="text-align:center;padding:60px 24px;">'
                        '<div style="font-size:54px">🛠️</div>'
                        '<h3 style="margin:8px 0 6px">Prêt pour le diagnostic</h3>'
                        f'<p style="color:{MUTED};margin:0">Renseignez l\'état de la machine puis '
                        'lancez le diagnostic pour obtenir le risque de panne.</p></div>',
                        unsafe_allow_html=True)
        else:
            p = float(model.predict_proba(encode(pd.DataFrame([entree]), mappings))[0, 1])
            alerte = p >= seuil
            color = DANGER if alerte else OK
            jauge = go.Figure(go.Indicator(
                mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 46, "color": color}},
                gauge={"axis": {"range": [0, 100], "tickcolor": MUTED}, "bar": {"color": color, "thickness": .3},
                       "bgcolor": "#F1F5F9", "borderwidth": 0,
                       "steps": [{"range": [0, seuil * 100], "color": "#ECFDF5"},
                                 {"range": [seuil * 100, 100], "color": "#FEF2F2"}],
                       "threshold": {"line": {"color": INK, "width": 3}, "value": seuil * 100}}))
            st.plotly_chart(style_fig(jauge, 260).update_layout(margin=dict(t=20, b=0)), width="stretch")

            if alerte:
                st.markdown(f'<div class="verdict" style="background:linear-gradient(120deg,{DANGER},#B91C1C)">'
                            '<h3>⚠️ Risque de panne élevé</h3><p>Planifier une inspection et une intervention '
                            'de maintenance rapidement.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict" style="background:linear-gradient(120deg,{OK},#047857)">'
                            '<h3>✅ Machine en bon état</h3><p>Aucune intervention urgente : poursuivre la '
                            'surveillance habituelle.</p></div>', unsafe_allow_html=True)

            hauts = [c for c in SENSORS if entree[c] > q90[c]]
            chips = "".join(f'<span class="chip up">▲ {LABELS[c]}</span>' for c in hauts) or \
                '<span class="chip">Tous les capteurs sont dans les valeurs habituelles</span>'
            st.markdown(f'<div class="card" style="margin-top:14px"><b>Capteurs à surveiller</b>'
                        f'<div style="margin-top:6px">{chips}</div>'
                        f'<p style="color:{MUTED};font-size:12.5px;margin:10px 0 0">▲ = valeur au-dessus de 90 % '
                        'de l\'historique.</p></div>', unsafe_allow_html=True)

# ─────────────────────────── Exploration ───────────────────────────
with tab_data:
    a, b = st.columns(2, gap="large")
    taux = df.groupby("machine_type")[TARGET].mean().sort_values().reset_index()
    fig = px.bar(taux, x=TARGET, y="machine_type", orientation="h", text_auto=".1%",
                 labels={TARGET: "Taux de panne", "machine_type": ""}, title="Taux de panne par machine")
    fig.update_traces(marker_color=ACCENT, marker_line_width=0)
    fig.update_xaxes(tickformat=".0%")
    a.plotly_chart(style_fig(fig), width="stretch")

    fam = df.groupby("part_family")[TARGET].mean().sort_values().reset_index()
    fig = px.bar(fam, x=TARGET, y="part_family", orientation="h", text_auto=".1%",
                 labels={TARGET: "Taux de panne", "part_family": ""}, title="Taux de panne par famille de pièce")
    fig.update_traces(marker_color="#0EA5E9", marker_line_width=0)
    fig.update_xaxes(tickformat=".0%")
    b.plotly_chart(style_fig(fig), width="stretch")

    capteur = st.selectbox("Comparer un capteur", SENSORS, format_func=lambda c: LABELS[c])
    ech = df.sample(min(len(df), 20000), random_state=1).copy()
    ech["État"] = ech[TARGET].map({0: "Normal", 1: "Panne"})
    fig = px.histogram(ech, x=capteur, color="État", barmode="overlay", nbins=60, opacity=.7,
                       color_discrete_map={"Normal": "#94A3B8", "Panne": DANGER},
                       labels={capteur: LABELS[capteur]}, title=f"{LABELS[capteur]} : panne vs normal")
    st.plotly_chart(style_fig(fig, 360), width="stretch")

# ─────────────────────────── Performance ───────────────────────────
with tab_model:
    res = {
        "Accuracy": accuracy_score(y_te, y_pred_te),
        "Précision pannes": precision_score(y_te, y_pred_te, zero_division=0),
        "Rappel pannes": recall_score(y_te, y_pred_te, zero_division=0),
        "F1 pannes": f1_score(y_te, y_pred_te, zero_division=0),
    }
    cols = st.columns(4)
    subs = ["prédictions correctes", "alertes justifiées", "pannes détectées", "équilibre précision / rappel"]
    for col, (k, v), s in zip(cols, res.items(), subs):
        kpi(col, k, f"{v:.1%}", s)
    st.write("")

    a, b = st.columns(2, gap="large")
    cm = confusion_matrix(y_te, y_pred_te)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, "#FFF7ED"], [1, ACCENT]],
                    x=["Prédit normal", "Prédit panne"], y=["Réel normal", "Réel panne"],
                    title="Matrice de confusion")
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textfont_size=18)
    a.plotly_chart(style_fig(fig), width="stretch")

    imp = pd.Series(model.feature_importances_, index=X_te.columns).sort_values().tail(8)
    fig = go.Figure(go.Bar(x=imp.values, y=[LABELS.get(i, i) for i in imp.index], orientation="h",
                           marker_color="#8B5CF6"))
    fig.update_layout(title="Variables les plus influentes")
    b.plotly_chart(style_fig(fig), width="stretch")

    st.caption(f"Modèle : Random Forest (200 arbres, classes équilibrées), évalué sur {len(y_te):,} cas de test. "
               .replace(",", " ") + "Baisser le seuil d'alerte permet de rater moins de pannes, au prix de "
               "fausses alertes supplémentaires.")
