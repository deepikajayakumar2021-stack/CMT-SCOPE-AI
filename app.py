import os
import glob
import re
import importlib.util

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

st.set_page_config(
    page_title="CMT-SCOPE AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = r"D:\CMT_PROJECT"

PROCESSED_DIR = os.path.join(
    PROJECT_DIR,
    "processed"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

MODELS_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

RAG_DIR = os.path.join(
    PROJECT_DIR,
    "rag"
)

MANIFEST_PATH = os.path.join(
    PROCESSED_DIR,
    "splits",
    "stage5c_patient_manifest.csv"
)

CLINICAL_PATH = os.path.join(
    PROCESSED_DIR,
    "clinical_baseline.csv"
)

TEST_PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)

DUAL_AXIS_PATH = os.path.join(
    RESULTS_DIR,
    "dual_axis_patient_profile.csv"
)

MULTI_GAIT_PATH = os.path.join(
    PROCESSED_DIR,
    "multi_gait_windows.npz"
)

SECONDARY_GAIT_PATH = os.path.join(
    PROCESSED_DIR,
    "secondary_gait_windows.npz"
)

COMBINED_GAIT_PATH = os.path.join(
    PROCESSED_DIR,
    "combined_gait_windows.npz"
)


CLASS_NAMES = [
    "Mild",
    "Moderate",
    "Severe"
]

SEVERITY_MAP = {
    0: "Mild",
    1: "Moderate",
    2: "Severe"
}


CHECKPOINT_OPTIONS = [
    os.path.join(
        MODELS_DIR,
        "best_cmt_severity_model.pth"
    ),
    os.path.join(
        MODELS_DIR,
        "best_cmt_model.pth"
    )
]

CHECKPOINT_PATH = None

for checkpoint in CHECKPOINT_OPTIONS:

    if os.path.exists(checkpoint):

        CHECKPOINT_PATH = checkpoint

        break

st.markdown(
    """
    <style>

    /* Main page */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* Main title */

    .main-title {
        font-size: 38px;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .main-subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 25px;
    }


    /* Section */

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }


    /* Cards */

    .card {
        padding: 20px;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        background: #ffffff;
        min-height: 125px;
    }

    .card-title {
        font-size: 15px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 29px;
        font-weight: 700;
    }

    .card-description {
        font-size: 13px;
        color: #6b7280;
        margin-top: 5px;
    }


    /* Research evidence */

    .evidence-card {
        padding: 18px;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #fafafa;
        margin-bottom: 15px;
    }


    /* LLM */

    .llm-section {
        padding: 18px;
        border-left: 4px solid #777;
        background: #fafafa;
        border-radius: 8px;
        margin-bottom: 15px;
    }


    body { background: linear-gradient(180deg,#f5f9ff 0%,#ffffff 42%,#f7fbff 100%); }
    .hero-wrap { display:flex; align-items:center; gap:34px; padding:28px 34px; border-radius:28px; background:linear-gradient(135deg,#eaf4ff 0%,#f4efff 55%,#e8fbf6 100%); border:1px solid #d8e7ff; box-shadow:0 16px 45px rgba(37,99,235,.10); margin-bottom:26px; }
    .hero-circle { width:230px; height:230px; min-width:230px; border-radius:50%; overflow:hidden; border:8px solid rgba(255,255,255,.88); box-shadow:0 12px 30px rgba(15,23,42,.16); background:#fff; }
    .hero-circle img { width:100%; height:100%; object-fit:cover; }
    .hero-copy { max-width:850px; }
    .hero-kicker { font-size:13px; font-weight:800; letter-spacing:1.8px; color:#2563eb; margin-bottom:7px; }
    .hero-title { font-size:48px; line-height:1.05; font-weight:900; color:#102a43; margin-bottom:8px; }
    .hero-subtitle { font-size:22px; font-weight:750; color:#334e68; margin-bottom:10px; }
    .hero-text { font-size:16px; color:#52606d; line-height:1.65; }

    /* Final medical dashboard theme */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #eef7ff 0%, #f7f3ff 48%, #effcf9 100%);
    }
    [data-testid="stHeader"] { background: rgba(255,255,255,0.65); }
    .block-container { max-width: 1450px; }
    .medical-hero {
        background: linear-gradient(135deg, rgba(35,110,190,.96), rgba(116,76,190,.94));
        border-radius: 26px; padding: 30px 34px; color: white;
        box-shadow: 0 16px 40px rgba(55,80,140,.18); margin-bottom: 24px;
    }
    .medical-hero h1 { font-size: 42px; font-weight: 850; margin: 0; }
    .medical-hero p { font-size: 17px; opacity: .94; margin: 8px 0 0; }
    .image-card { background: rgba(255,255,255,.88); border-radius: 24px; padding: 14px; box-shadow: 0 12px 30px rgba(50,70,120,.12); }
    .page-card { background: rgba(255,255,255,.82); border: 1px solid rgba(120,140,180,.16); border-radius: 20px; padding: 18px; box-shadow: 0 8px 25px rgba(50,70,120,.08); }
    .mini-title { font-size: 18px; font-weight: 800; color: #243b6b; }
    .eval-banner {
        background: linear-gradient(135deg, #dbeafe 0%, #ede9fe 52%, #ccfbf1 100%);
        border: 1px solid #c7d2fe; border-radius: 22px; padding: 22px 26px;
        margin: 8px 0 22px; box-shadow: 0 10px 28px rgba(37,99,235,.10);
    }
    .eval-banner-title { font-size: 28px; font-weight: 850; color: #172554; margin-bottom: 5px; }
    .eval-banner-text { color: #475569; font-size: 15px; line-height: 1.55; }
    .chart-card { background: rgba(255,255,255,.88); border: 1px solid rgba(99,102,241,.12); border-radius: 18px; padding: 8px 10px 2px; box-shadow: 0 7px 22px rgba(30,41,59,.07); }

    /* Footer */

    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        padding-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

def load_csv(path):

    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)

    except Exception as e:

        st.error(
            f"Unable to read {os.path.basename(path)}: {e}"
        )

        return None


def normalize_column_name(name):

    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def find_column(df, candidates):

    if df is None:
        return None

    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:

        key = normalize_column_name(candidate)

        if key in normalized:

            return normalized[key]

    return None


def severity_from_score(score):

    try:

        score = float(score)

        if score <= 6:
            return "Mild"

        if score <= 16:
            return "Moderate"

        return "Severe"

    except Exception:

        return "Unknown"


def normalize_prediction(value):

    text = str(value).strip()

    lower = text.lower()

    if lower in [
        "mild",
        "moderate",
        "severe"
    ]:

        return lower.title()

    try:

        number = int(float(value))

        if number in SEVERITY_MAP:

            return SEVERITY_MAP[number]

    except Exception:

        pass

    return value


def is_severity_series(series):

    if series is None:
        return False

    values = set(
        str(x).strip().lower()
        for x in series.dropna().unique()
    )

    return (
        len(values) > 0
        and values.issubset(
            {
                "mild",
                "moderate",
                "severe"
            }
        )
    )


def format_percent(value):

    try:

        value = float(value)

        if value <= 1:

            value *= 100

        return f"{value:.1f}%"

    except Exception:

        return "N/A"



@st.cache_data(show_spinner=False)
def load_manifest():

    if not os.path.exists(MANIFEST_PATH):

        return None

    df = pd.read_csv(
        MANIFEST_PATH
    )

    if (
        "severity_name" not in df.columns
        and "Cmtes_score" in df.columns
    ):

        df["severity_name"] = (
            df["Cmtes_score"]
            .apply(severity_from_score)
        )

    return df


@st.cache_data(show_spinner=False)
def load_test_predictions():

    return load_csv(
        TEST_PREDICTIONS_PATH
    )


@st.cache_data(show_spinner=False)
def load_clinical():

    return load_csv(
        CLINICAL_PATH
    )


@st.cache_data(show_spinner=False)
def load_dual_axis():

    return load_csv(
        DUAL_AXIS_PATH
    )


manifest = load_manifest()

test_predictions = load_test_predictions()

clinical_df = load_clinical()

dual_axis_df = load_dual_axis()



# SUPPLEMENTARY GAIT SUMMARY


@st.cache_data(show_spinner=False)
def get_gait_summary():

    result = {
        "multi_users": 0,
        "multi_recordings": 0,
        "multi_windows": 0,
        "secondary_windows": 0,
        "combined_windows": 0
    }


    # --------------------------------------------------------
    # Multi-sensor dataset
    # --------------------------------------------------------

    if os.path.exists(
        MULTI_GAIT_PATH
    ):

        try:

            with np.load(
                MULTI_GAIT_PATH,
                allow_pickle=True
            ) as data:

                gait = data["gait"]

                result["multi_windows"] = int(
                    len(gait)
                )

                if "user_id" in data:

                    result["multi_users"] = int(
                        len(
                            np.unique(
                                data["user_id"]
                            )
                        )
                    )

                if "recording_id" in data:

                    result["multi_recordings"] = int(
                        len(
                            np.unique(
                                data["recording_id"]
                            )
                        )
                    )

        except Exception:
            pass


    # --------------------------------------------------------
    # Secondary dataset
    # --------------------------------------------------------

    if os.path.exists(
        SECONDARY_GAIT_PATH
    ):

        try:

            with np.load(
                SECONDARY_GAIT_PATH,
                allow_pickle=True
            ) as data:

                result["secondary_windows"] = int(
                    len(data["X"])
                )

        except Exception:
            pass


    # --------------------------------------------------------
    # Combined dataset
    # --------------------------------------------------------

    if os.path.exists(
        COMBINED_GAIT_PATH
    ):

        try:

            with np.load(
                COMBINED_GAIT_PATH,
                allow_pickle=True
            ) as data:

                result["combined_windows"] = int(
                    len(data["X"])
                )

        except Exception:
            pass


    if result["combined_windows"] == 0:

        result["combined_windows"] = (
            result["multi_windows"]
            +
            result["secondary_windows"]
        )

    return result


gait_summary = get_gait_summary()



# EXPLAINABILITY FILES


def get_explanation_patients():

    patients = set()

    patterns = [
        "*_clinical_shap.csv",
        "*_gait_temporal_explanation.csv",
        "*_adaptive_fusion.csv",
        "*_baseline_explanation.csv"
    ]

    for pattern in patterns:

        for path in glob.glob(
            os.path.join(
                RESULTS_DIR,
                pattern
            )
        ):

            name = os.path.basename(
                path
            )

            patient = name.split("_")[0]

            if patient:

                patients.add(patient)

    return sorted(patients)


explanation_patients = (
    get_explanation_patients()
)



# PATIENT-LINKED GAIT / EXPLAINABILITY HELPERS


def patient_ids_from_manifest():
    if manifest is None:
        return []
    id_col = find_column(manifest, ["ID", "Patient_ID", "Patient"])
    if not id_col:
        return []
    return sorted(manifest[id_col].dropna().astype(str).unique().tolist())


def find_patient_gait_file(patient_id):
    """Find the patient's own CMT1A gait file. Never substitute another patient."""
    candidates = []
    if manifest is not None:
        id_col = find_column(manifest, ["ID", "Patient_ID", "Patient"])
        gait_col = find_column(manifest, ["gait_file", "gait path", "gait_path"])
        if id_col and gait_col:
            rows = manifest[manifest[id_col].astype(str) == str(patient_id)]
            if not rows.empty:
                value = str(rows.iloc[0].get(gait_col, "")).strip()
                if value and value.lower() != "nan":
                    candidates.append(value)
                    candidates.append(os.path.join(PROJECT_DIR, value))
                    candidates.append(os.path.join(PROCESSED_DIR, value))

    candidates += [
        os.path.join(PROCESSED_DIR, "gait_windows", f"{patient_id}_gait.npz"),
        os.path.join(PROCESSED_DIR, "gait_windows", f"{str(patient_id).replace('-', '_')}_gait.npz"),
        os.path.join(PROCESSED_DIR, "gait_windows", f"{str(patient_id).replace('_', '-')}_gait.npz"),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    return None


@st.cache_data(show_spinner=False)
def load_patient_gait(patient_id):
    path = find_patient_gait_file(patient_id)
    if not path:
        return None, None
    try:
        with np.load(path, allow_pickle=True) as gd:
            key = next((k for k in ["gait", "X", "data", "windows"] if k in gd.files), None)
            if key is None:
                return None, path
            arr = np.asarray(gd[key])
            if arr.ndim == 2:
                arr = arr[None, ...]
            if arr.ndim != 3:
                return None, path
            # Standard project representation is [windows, channels, samples].
            if arr.shape[1] <= 6 and arr.shape[2] > arr.shape[1]:
                arr = np.transpose(arr, (0, 2, 1))
            return arr.astype(np.float32), path
    except Exception:
        return None, path


def build_clinical_fallback(patient_id):
    """Cohort-normalized clinical contribution when a saved SHAP file is absent.
    This is explicitly not labelled SHAP."""
    if manifest is None:
        return pd.DataFrame()
    row = manifest[manifest["ID"].astype(str) == str(patient_id)]
    if row.empty:
        return pd.DataFrame()
    numeric_candidates = ["Age (Years)", "Pinprick", "Vibration", "Cmtes_score", "CMT-FOM_Total", "MMT_average"]
    rows=[]
    for col in numeric_candidates:
        if col not in manifest.columns:
            continue
        vals=pd.to_numeric(manifest[col], errors="coerce")
        x=pd.to_numeric(row.iloc[0][col], errors="coerce")
        if pd.isna(x) or vals.dropna().empty:
            continue
        mean=float(vals.mean()); std=float(vals.std(ddof=0))
        z=0.0 if std==0 or np.isnan(std) else float((x-mean)/std)
        rows.append({"Feature": col, "Patient Value": float(x), "Cohort Mean": round(mean,3), "Standardized Difference": round(z,3), "Relative Contribution": round(abs(z),3)})
    return pd.DataFrame(rows).sort_values("Relative Contribution", ascending=False) if rows else pd.DataFrame()


def build_temporal_fallback(patient_id):
    """Patient-specific signal analysis used only when a saved temporal attribution file is absent."""
    arr, path = load_patient_gait(patient_id)
    if arr is None or len(arr)==0:
        return pd.DataFrame(), path
    signal=np.nanmean(arr, axis=0)
    if signal.ndim != 2:
        return pd.DataFrame(), path
    z=np.abs((signal-np.nanmean(signal,axis=0))/(np.nanstd(signal,axis=0)+1e-8))
    saliency=np.nanmean(z,axis=1)
    mean_signal=np.nanmean(signal,axis=1)
    return pd.DataFrame({"Sample":np.arange(len(saliency)),"Temporal Importance":saliency,"Mean Gait Signal":mean_signal}), path



# PATIENT CLINICAL + GAIT HELPERS


def patient_row_with_clinical(patient_id):
    """Return one patient row with clinical values resolved from manifest or clinical_baseline."""
    base = pd.DataFrame()
    if manifest is not None:
        idc = find_column(manifest, ["ID", "Patient_ID", "Patient"])
        if idc:
            base = manifest[manifest[idc].astype(str) == str(patient_id)].copy()
    if base.empty:
        base = pd.DataFrame([{"ID": patient_id}])
    else:
        base = base.iloc[[0]].copy()

    if clinical_df is not None and not clinical_df.empty:
        cid = find_column(clinical_df, ["ID", "Patient_ID", "Patient", "Subject_ID"])
        if cid:
            crow = clinical_df[clinical_df[cid].astype(str) == str(patient_id)]
            if not crow.empty:
                cr = crow.iloc[0]
                for col in clinical_df.columns:
                    if col not in base.columns or pd.isna(base.iloc[0].get(col)):
                        base.loc[base.index[0], col] = cr.get(col)
    return base


def clinical_value(row, candidates):
    if row is None or row.empty:
        return None
    c = find_column(row, candidates)
    if c is None:
        return None
    v = row.iloc[0].get(c)
    if pd.isna(v) or str(v).strip().lower() in {"nan", "none", ""}:
        return None
    return v


def gait_window_summary(patient_id):
    arr, path = load_patient_gait(patient_id)
    if arr is None or len(arr) == 0:
        return {"windows": 0, "samples": 0, "channels": 0, "path": path, "array": None}
    return {"windows": int(arr.shape[0]), "samples": int(arr.shape[1]), "channels": int(arr.shape[2]), "path": path, "array": arr}


def render_patient_gait(patient_id, key_prefix):
    info = gait_window_summary(patient_id)
    arr = info["array"]
    if arr is None:
        st.warning(f"No processed gait recording is linked to {patient_id}.")
        return
    st.markdown("#### Patient-linked gait signal")
    m1,m2,m3 = st.columns(3)
    m1.metric("Gait Windows", info["windows"])
    m2.metric("Samples / Window", info["samples"])
    m3.metric("Channels", info["channels"])
    max_windows = min(info["windows"], 50)
    selected_window = st.slider("Gait window", 0, max_windows-1, max_windows//2, key=f"{key_prefix}_window")
    window = arr[selected_window]
    channel_names = ["Accel X", "Accel Y", "Accel Z"][:window.shape[1]]
    if plotly_available():
        fig = go.Figure()
        for j,n in enumerate(channel_names):
            fig.add_trace(go.Scatter(x=np.arange(window.shape[0]), y=window[:,j], mode="lines", name=n))
        fig.update_layout(title=f"{patient_id} — Gait Window {selected_window+1}", xaxis_title="Sample", yaxis_title="Acceleration", height=390)
        st.plotly_chart(fig, use_container_width=True)
        mean_signal = np.nanmean(arr, axis=0)
        fig2 = go.Figure()
        for j,n in enumerate(channel_names):
            fig2.add_trace(go.Scatter(x=np.arange(mean_signal.shape[0]), y=mean_signal[:,j], mode="lines", name=n))
        fig2.update_layout(title=f"{patient_id} — Mean Across {info["windows"]} Gait Windows", xaxis_title="Sample", yaxis_title="Mean Signal", height=360)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.line_chart(pd.DataFrame(window, columns=channel_names))
    st.caption(f"Source: {os.path.basename(info["path"])} · This is the selected patient's own gait data.")


# ============================================================
# LLM FILES
# ============================================================

def get_llm_files():

    return sorted(
        glob.glob(
            os.path.join(
                RESULTS_DIR,
                "*_llm_explanation.txt"
            )
        )
    )


llm_files = get_llm_files()



# RAG ENGINE


def load_rag_engine():

    path = os.path.join(
        PROJECT_DIR,
        "rag_engine.py"
    )

    if not os.path.exists(path):

        return None

    try:

        spec = (
            importlib.util
            .spec_from_file_location(
                "rag_engine",
                path
            )
        )

        module = (
            importlib.util
            .module_from_spec(
                spec
            )
        )

        spec.loader.exec_module(
            module
        )

        return module

    except Exception:

        return None


rag_engine = load_rag_engine()



def clean_llm_text(text):

    if not text:

        return ""

    text = str(text)

    # Convert escaped Markdown
    text = text.replace(
        r"\*\*",
        "**"
    )

    text = text.replace(
        r"\*",
        "*"
    )

    # Remove accidental code fences
    text = text.replace(
        "```markdown",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    return text.strip()



def plotly_available():
    return px is not None and go is not None


def extract_important_evidence(text_value, max_sentences=2, max_chars=520):
    """Extract the most clinically informative findings from one retrieved passage."""
    text_value = re.sub(r"\s+", " ", str(text_value or "")).strip()
    if not text_value:
        return "No evidence text available."

    # Split conservatively so abbreviations and decimal values are not destroyed.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text_value)
    sentences = [x.strip(" -•") for x in sentences if len(x.strip()) >= 40]

    high_value_terms = {
        "cmt1a": 5, "charcot-marie-tooth": 4, "severity": 4,
        "gait": 4, "walking": 3, "cmt-fom": 5, "cmt neuropathy score": 5,
        "clinical outcome": 4, "functional": 3, "impairment": 4,
        "muscle weakness": 4, "foot drop": 4, "balance": 3,
        "step length": 4, "gait speed": 4, "stride": 3,
        "fatigue": 3, "correlation": 3, "significant": 2,
        "associated": 2, "predict": 2, "reliab": 2, "result": 2,
        "conclusion": 2, "finding": 2
    }

    scored = []
    for idx, sentence in enumerate(sentences):
        lower = sentence.lower()
        score = 0.0
        for term, weight in high_value_terms.items():
            if term in lower:
                score += weight
        if any(word in lower for word in ["found", "showed", "demonstrated", "associated", "correlated", "significant", "concluded"]):
            score += 2.5
        # Prefer substantive findings over very short/background sentences.
        score += min(len(sentence) / 220.0, 1.5)
        scored.append((score, idx, sentence))

    ranked = sorted(scored, key=lambda x: (-x[0], x[1]))
    selected = [item[2] for item in ranked[:max_sentences]]
    if not selected:
        selected = sentences[:max_sentences] or [text_value]

    # Keep the evidence readable in original paper order.
    selected = sorted(selected, key=lambda x: text_value.find(x))
    result = " ".join(selected)
    if len(result) > max_chars:
        result = result[:max_chars - 3].rsplit(" ", 1)[0] + "..."
    return result


def patient_value(df, patient_id, candidates):
    """Return the first available patient value from candidate columns."""
    if df is None or df.empty:
        return None
    id_col = find_column(df, ["ID", "Patient_ID", "Patient", "patient_id"])
    if not id_col:
        return None
    rows = df[df[id_col].astype(str) == str(patient_id)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    col = find_column(df, candidates)
    if col and pd.notna(row.get(col)):
        return row.get(col)
    return None


def build_patient_evidence_query(patient_id):
    """Build a patient-specific retrieval query without inventing unavailable measurements."""
    parts = ["CMT1A clinical severity gait impairment functional outcome evidence"]
    if manifest is not None:
        row = patient_value(manifest, patient_id, ["severity_name"])
        if row:
            parts.append(f"severity {row}")
        for label, candidates in [
            ("CMT examination score", ["Cmtes_score", "CMTES"]),
            ("CMT functional outcome", ["CMT-FOM_Total", "CMTFOM_Total", "CMT_FOM_Total"]),
            ("muscle strength", ["MMT_average", "MMT average"]),
            ("age", ["Age"]),
        ]:
            value = patient_value(manifest, patient_id, candidates)
            if value is not None:
                parts.append(f"{label} {value}")
    # Explainability outputs can add the patient-specific concepts actually present in the project.
    shap_path = os.path.join(RESULTS_DIR, f"{patient_id}_clinical_shap.csv")
    if os.path.exists(shap_path):
        sdf = load_csv(shap_path)
        if sdf is not None and not sdf.empty:
            feature_col = find_column(sdf, ["feature", "Feature", "name", "variable"])
            if feature_col:
                features = sdf[feature_col].dropna().astype(str).head(5).tolist()
                if features:
                    parts.append("clinical features " + ", ".join(features))
    return " ".join(parts)


def render_evidence_cards(results, patient_id=None):
    """Render retrieved evidence as readable evidence blocks with source + finding."""
    for index, item in enumerate(results, start=1):
        source = str(item.get("source", "Unknown source"))
        score = item.get("score", 0)
        try:
            score_text = f"{float(score):.4f}"
        except Exception:
            score_text = "N/A"
        finding = extract_important_evidence(item.get("text", ""), max_sentences=2, max_chars=560)
        label = f"Evidence {index}"
        patient_label = f" · {patient_id}" if patient_id else ""
        st.markdown(
            f"""
            <div class="evidence-card">
                <div style="font-size:12px;color:#64748b;font-weight:700;">{label}{patient_label}</div>
                <div style="font-size:16px;font-weight:700;margin:5px 0 8px;">{source}</div>
                <div style="font-size:13px;color:#64748b;margin-bottom:8px;">Retrieval similarity: {score_text}</div>
                <div style="font-size:14px;line-height:1.6;"><b>Important evidence:</b> {finding}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
   
    
def render_actual_predicted_scatter(actual, predicted, patient_labels=None):
    """Render patient-level actual-vs-predicted severity as a real scatter graph."""
    if actual is None or predicted is None or len(actual) == 0:
        return

    severity_to_num = {"Mild": 1, "Moderate": 2, "Severe": 3}
    labels = patient_labels if patient_labels is not None else [f"Patient {i+1}" for i in range(len(actual))]

    df = pd.DataFrame({
        "Patient": [str(x) for x in labels],
        "Actual": actual.astype(str).values,
        "Predicted": predicted.astype(str).values
    })
    df["Actual Level"] = df["Actual"].map(severity_to_num)
    df["Predicted Level"] = df["Predicted"].map(severity_to_num)
    df["Result"] = np.where(df["Actual"] == df["Predicted"], "Correct", "Misclassified")
    df["Point Label"] = df.apply(
        lambda r: f"{r['Patient']}<br>A: {r['Actual']}<br>P: {r['Predicted']}", axis=1
    )

    if not plotly_available():
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    st.caption(
        "Each dot is one evaluated patient. The diagonal line represents a correct prediction; "
        "points away from it are prediction errors."
    )

    fig = go.Figure()

    # Ideal diagonal.
    fig.add_trace(go.Scatter(
        x=[1, 2, 3], y=[1, 2, 3], mode="lines",
        name="Correct prediction line",
        line=dict(color="#94a3b8", width=3, dash="dash"),
        hoverinfo="skip"
    ))

    for result, color, symbol in [
        ("Correct", "#10b981", "circle"),
        ("Misclassified", "#ef4444", "x")
    ]:
        d = df[df["Result"] == result]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["Actual Level"],
            y=d["Predicted Level"],
            mode="markers+text",
            name=result,
            text=d["Patient"],
            textposition="top center",
            marker=dict(size=18, color=color, symbol=symbol, line=dict(color="white", width=2)),
            customdata=np.stack([d["Patient"], d["Actual"], d["Predicted"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Actual: %{customdata[1]}<br>"
                "Predicted: %{customdata[2]}<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Patient-Level Actual vs Predicted Severity",
        xaxis=dict(
            title="Actual Severity",
            tickmode="array", tickvals=[1,2,3], ticktext=CLASS_NAMES,
            range=[0.7,3.3], gridcolor="#e2e8f0", zeroline=False
        ),
        yaxis=dict(
            title="Predicted Severity",
            tickmode="array", tickvals=[1,2,3], ticktext=CLASS_NAMES,
            range=[0.7,3.3], gridcolor="#e2e8f0", zeroline=False
        ),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,.95)",
        font=dict(family="Arial", size=14),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig, use_container_width=True, key="actual_predicted_scatter_graph")

    # Patient-by-patient table immediately below the graph.
    display_df = df[["Patient", "Actual", "Predicted", "Result"]].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def render_metric_chart(metric_df):
    if not plotly_available() or metric_df.empty:
        st.dataframe(metric_df, use_container_width=True, hide_index=True)
        return
    long_df = metric_df.melt(id_vars="Class", value_vars=["Precision", "Recall", "F1 Score"], var_name="Metric", value_name="Score")
    fig = px.bar(long_df, x="Class", y="Score", color="Metric", barmode="group", range_y=[0, 1],
                 title="Per-Class Precision, Recall and F1 Score", text_auto=".2f")
    fig.update_layout(height=430, yaxis_title="Score", xaxis_title="Severity")
    st.plotly_chart(fig, use_container_width=True)


def render_distribution_chart(series, title, y_title="Patients"):
    data = pd.DataFrame({"Severity": CLASS_NAMES, "Count": series.reindex(CLASS_NAMES, fill_value=0).values})
    if not plotly_available():
        st.bar_chart(data.set_index("Severity"))
        return
    fig = px.bar(data, x="Severity", y="Count", color="Severity", text="Count", title=title,
                 color_discrete_sequence=["#3b82f6", "#8b5cf6", "#f59e0b"])
    fig.update_traces(textposition="outside", marker_line_width=1)
    fig.update_layout(height=390, yaxis_title=y_title, xaxis_title="Severity",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.55)")
    st.plotly_chart(fig, use_container_width=True)


# SIDEBAR
# ============================================================

st.sidebar.title(
    "CMT-SCOPE AI"
)

st.sidebar.caption(
    "Research Decision-Support Prototype"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Patient Analysis",
        "Severity Prediction",
        "Evaluation & Graphs",
        "Adaptive Fusion",
        "Dual-Axis Analysis",
        "Explainable AI",
        "Research & Evidence",
        "Model Results"
    ]
)

st.sidebar.markdown("---")

if st.sidebar.button(
    "Refresh Data"
):

    st.cache_data.clear()

    st.rerun()


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "Dashboard":
    st.markdown("""
    <div class="medical-hero">
      <h1>CMT-SCOPE AI</h1>
      <p>Explainable AI-Based CMT1A Disease Severity Prediction & Patient Gait Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    hero_left, hero_right = st.columns([0.85, 2.15], gap="large")
    with hero_left:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.image("https://commons.wikimedia.org/wiki/Special:FilePath/Charcot-marie-tooth%20foot.jpg", use_container_width=True)
        st.caption("Charcot–Marie–Tooth disease visual")
        st.markdown('</div>', unsafe_allow_html=True)
    with hero_right:
        st.markdown("### CMT1A Clinical Intelligence")
        st.write("Patient-linked clinical data and gait time-series are analysed together at the patient level. Gait windows are used for temporal representation learning and are never treated as separate patients.")
        patient_count = len(patient_ids_from_manifest())
        severity_counts = manifest["severity_name"].value_counts().reindex(CLASS_NAMES, fill_value=0) if manifest is not None and "severity_name" in manifest.columns else pd.Series(0,index=CLASS_NAMES)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("CMT1A Patients", patient_count)
        c2.metric("Gait Windows", gait_summary["combined_windows"])
        c3.metric("Mild", int(severity_counts.get("Mild",0)))
        c4.metric("Moderate", int(severity_counts.get("Moderate",0)))

    st.markdown("### Clinical Cohort Analysis")
    cc1,cc2 = st.columns(2)
    with cc1:
        render_distribution_chart(severity_counts, "CMT1A Severity Distribution")
    with cc2:
        if manifest is not None:
            ages = pd.to_numeric(manifest.get("Age (Years)"), errors="coerce") if "Age (Years)" in manifest.columns else pd.Series(dtype=float)
            if ages.notna().any() and plotly_available():
                age_df=pd.DataFrame({"Patient":manifest["ID"].astype(str),"Age":ages,"Severity":manifest["severity_name"]}).dropna(subset=["Age"])
                fig=px.bar(age_df,x="Patient",y="Age",color="Severity",title="Patient Age by Severity",text="Age")
                fig.update_layout(height=390,xaxis_title="Patient",yaxis_title="Age (Years)")
                st.plotly_chart(fig,use_container_width=True)
            else:
                pass

    st.markdown("### Patient-linked gait overview")
    if manifest is not None:
        rows=[]
        for pid in patient_ids_from_manifest():
            info=gait_window_summary(pid)
            row=patient_row_with_clinical(pid).iloc[0]
            rows.append({"Patient":pid,"Severity":row.get("severity_name","N/A"),"Age":clinical_value(pd.DataFrame([row]),["Age (Years)","Age","AgeYears"]) or "Not available","Gait Windows":info["windows"]})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    st.markdown("### What the system analyses")
    x,y,z=st.columns(3)
    x.markdown("**Clinical profile**")
    x.caption("Age, gender, pinprick, vibration and CMT-related clinical scores.")
    y.markdown("**Gait profile**")
    y.caption("Patient-specific temporal acceleration windows and gait patterns.")
    z.markdown("**Explainable prediction**")
    z.caption("Clinical contribution + gait temporal attribution for every CMT1A patient.")


# PAGE 2 — PATIENT ANALYSIS
# ============================================================

elif page == "Patient Analysis":
    st.markdown("## Patient Analysis")
    st.caption("All 15 CMT1A patients are available. Clinical information and that patient's own gait recording are shown together.")
    patient_ids=patient_ids_from_manifest()
    if not patient_ids:
        st.error("No CMT1A patients found.")
    else:
        patient_id=st.selectbox("Select CMT1A patient",patient_ids,key="patient_analysis_selector")
        prow=patient_row_with_clinical(patient_id)
        row=prow.iloc[0]
        info=gait_window_summary(patient_id)
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Patient",patient_id)
        c2.metric("Age",clinical_value(prow,["Age (Years)","Age","AgeYears"]) or "N/A")
        c3.metric("Severity",str(row.get("severity_name","N/A")))
        c4.metric("Gait Windows",info["windows"])
        c5.metric("Split",str(row.get("split","N/A")).title())

        st.markdown("### Clinical Profile")
        clinical_fields=[("Age",["Age (Years)","Age","AgeYears"]),("Gender",["Gender","Sex"]),("Pinprick",["Pinprick"]),("Vibration",["Vibration"]),("CMTES",["Cmtes_score","CMTES"]),("CMT-FOM",["CMT-FOM_Total"]),("MMT Average",["MMT_average"])]
        cdf=pd.DataFrame([{"Feature":label,"Patient Value":clinical_value(prow,cands) or "N/A"} for label,cands in clinical_fields])
        st.dataframe(cdf,use_container_width=True,hide_index=True)

        render_patient_gait(patient_id, "patient_analysis")

        st.markdown("### Patient Prediction")
        if test_predictions is not None:
            pc=find_column(test_predictions,["Patient_ID","ID","Patient"])
            if pc:
                result=test_predictions[test_predictions[pc].astype(str)==patient_id].copy()
                if not result.empty:
                    st.dataframe(result,use_container_width=True,hide_index=True)
                else:
                    st.caption("Prediction result is not available for this patient in the saved evaluation output.")


#  — SEVERITY PREDICTION


elif page == "Severity Prediction":

    st.header(
        "Severity Prediction"
    )

    st.write(
        "The table below shows the patient-level predictions "
        "generated during held-out evaluation."
    )

    if test_predictions is None:

        st.error(
            "test_predictions.csv is unavailable."
        )

    else:

        patient_column = find_column(
            test_predictions,
            [
                "Patient_ID",
                "ID",
                "Patient"
            ]
        )

        actual_column = find_column(
            test_predictions,
            [
                "Actual_CMT_FOM",
                "Actual_Severity",
                "Actual",
                "True_Label"
            ]
        )

        predicted_column = find_column(
            test_predictions,
            [
                "Predicted_CMT_FOM",
                "Predicted_Severity",
                "Predicted",
                "Prediction"
            ]
        )

        alpha_column = find_column(
            test_predictions,
            [
                "Adaptive_Alpha",
                "Alpha"
            ]
        )


        # ----------------------------------------------------
        # Patient table
        # ----------------------------------------------------

        prediction_table = pd.DataFrame()

        if patient_column:

            prediction_table[
                "Patient"
            ] = test_predictions[
                patient_column
            ]

        if actual_column:

            prediction_table[
                "Actual Output"
            ] = test_predictions[
                actual_column
            ]

        if predicted_column:

            prediction_table[
                "Predicted Output"
            ] = test_predictions[
                predicted_column
            ]

        if alpha_column:

            prediction_table[
                "Adaptive Alpha"
            ] = test_predictions[
                alpha_column
            ]


        st.dataframe(
            prediction_table,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # Severity labels
        # ----------------------------------------------------

        predicted_severity = None
        actual_severity = None

        if predicted_column:

            predicted_severity = (
                test_predictions[
                    predicted_column
                ]
                .apply(
                    normalize_prediction
                )
            )

        if actual_column:

            actual_severity = (
                test_predictions[
                    actual_column
                ]
                .apply(
                    normalize_prediction
                )
            )


        if (
            predicted_severity is not None
            and is_severity_series(
                predicted_severity
            )
        ):

            st.markdown(
                "### Prediction Distribution"
            )

            distribution = (
                predicted_severity
                .value_counts()
                .reindex(
                    CLASS_NAMES,
                    fill_value=0
                )
            )

            st.bar_chart(
                distribution
            )

            


            # ------------------------------------------------
            # Important interpretation
            # ------------------------------------------------

            if (
                len(predicted_severity) > 0
                and
                (
                    predicted_severity
                    == "Mild"
                ).all()
            ):

                st.info(
                    "In the current held-out test set, all "
                    "evaluated patients were predicted as Mild. "
                    "This describes the current model output for "
                    "the test set; it does not mean that all "
                    "CMT1A patients are Mild."
                )


# ============================================================
# PAGE 4 — EVALUATION & GRAPHS
# ============================================================

elif page == "Evaluation & Graphs":

    st.markdown("""
    <div class="eval-banner">
      <div class="eval-banner-title">Evaluation & Interactive Graphs</div>
      <div class="eval-banner-text">Patient-level performance, confusion patterns, severity distribution, prediction behaviour, adaptive fusion and gait analytics — presented as reviewer-friendly visual evidence.</div>
    </div>
    """, unsafe_allow_html=True)

    if test_predictions is None:
        st.error("test_predictions.csv is unavailable.")
    else:
        patient_column = find_column(test_predictions, ["Patient_ID", "ID", "Patient"])
        actual_column = find_column(test_predictions, ["Actual_CMT_FOM", "Actual_Severity", "Actual", "True_Label"])
        predicted_column = find_column(test_predictions, ["Predicted_CMT_FOM", "Predicted_Severity", "Predicted", "Prediction"])
        alpha_column = find_column(test_predictions, ["Adaptive_Alpha", "Alpha"])
        confidence_column = find_column(test_predictions, ["Confidence", "Prediction_Confidence", "Max_Probability", "Probability", "Predicted_Probability"])

        actual = test_predictions[actual_column].apply(normalize_prediction) if actual_column else None
        predicted = test_predictions[predicted_column].apply(normalize_prediction) if predicted_column else None

        if actual is None or predicted is None:
            st.warning("Actual and predicted severity columns are required to draw the evaluation graphs.")
        else:
            # ----------------------------------------------------
            # KPI strip
            # ----------------------------------------------------
            accuracy = float((actual.values == predicted.values).mean())
            recalls = []
            precisions = []
            f1s = []
            rows = []
            for cls in CLASS_NAMES:
                tp = int(((actual == cls) & (predicted == cls)).sum())
                fp = int(((actual != cls) & (predicted == cls)).sum())
                fn = int(((actual == cls) & (predicted != cls)).sum())
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                rows.append([cls, precision, recall, f1])
                precisions.append(precision); recalls.append(recall); f1s.append(f1)
            metric_df = pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1 Score"])
           
            macro_f1 = float(np.mean(f1s))

            k1,k2 = st.columns(2)
           
            k1.metric("Macro F1", f"{macro_f1*100:.1f}%")
            k2.metric("Evaluated Patients", len(test_predictions))

            if "Severe" not in set(actual):
                st.caption("Note: the current held-out test set contains no Severe patient, so Severe-class test performance is not estimable from this split.")

            
            # ----------------------------------------------------
            # 2. Per-class performance
            

            baseline_path = os.path.join(
                r"D:\CMT_PROJECT",
                "results",
                "baseline_comparison_updated",
                "baseline_metrics.csv"
            )

            if os.path.exists(baseline_path):
                baseline_df = pd.read_csv(baseline_path)

                model_col = (
                    "Model" if "Model" in baseline_df.columns
                    else None
                )

                if model_col is not None:
                    best_row = baseline_df[
                        baseline_df[model_col].astype(str)
                        == "Extra Trees - More Features"
                    ]

                    if not best_row.empty:
                        best = best_row.iloc[0]
                        b1 = st.columns(1)[0]

                        b1.metric(
                        "Baseline Macro F1",
                        f"{float(best['Macro F1 (%)']):.2f}%"
                        )
                        
                    else:
                        st.warning(
                            "Extra Trees - More Features was not "
                            "found in the baseline results CSV."
                        )
                else:
                    st.warning(
                        "The baseline CSV does not contain a Model column."
                    )
            else:
                st.info(
                    "Baseline results not found. Run "
                    "scripts\\compare_baseines.py first."
                )
            
            st.markdown("### 1. Per-Class Performance")
            if plotly_available():
                long_metrics = metric_df.melt(id_vars="Class", value_vars=["Precision","Recall","F1 Score"], var_name="Metric", value_name="Score")
                fig = px.bar(long_metrics, x="Class", y="Score", color="Metric", barmode="group", text=long_metrics["Score"].map(lambda x:f"{x:.2f}"),
                             color_discrete_sequence=["#4f46e5","#0d9488","#f59e0b"], title="Precision, Recall and F1 by Severity")
                fig.update_yaxes(range=[0,1], title="Score"); fig.update_xaxes(title="Severity")
                fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.55)")
                st.plotly_chart(fig, use_container_width=True, key="per_class_metric_graph")
            st.dataframe(metric_df.style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1 Score":"{:.3f}"}), use_container_width=True, hide_index=True)

            # ----------------------------------------------------
            # 2. Actual vs predicted — patient-level scatter graph
            # ----------------------------------------------------
            st.markdown("### 2. Actual vs Predicted Severity")
            patient_labels = (
                test_predictions[patient_column].astype(str).tolist()
                if patient_column
                else [f"Patient {i+1}" for i in range(len(test_predictions))]
            )
            render_actual_predicted_scatter(actual, predicted, patient_labels)

            # ----------------------------------------------------
            # 3. Full cohort severity distribution
            # ----------------------------------------------------
            st.markdown("### 3. CMT1A Cohort Severity Distribution")
            if manifest is not None and "severity_name" in manifest.columns:
                sev=manifest["severity_name"].value_counts().reindex(CLASS_NAMES,fill_value=0)
                render_distribution_chart(sev,"All CMT1A Patients by Severity")

            # ----------------------------------------------------
            # 4. Prediction distribution
            # ----------------------------------------------------
            st.markdown("### 4. Model Prediction Distribution")
            pdist=predicted.value_counts().reindex(CLASS_NAMES,fill_value=0)
            render_distribution_chart(pdist,"Held-Out Model Predictions")

            
            # 5. Adaptive fusion
            
            st.markdown("### 5. Adaptive Fusion — Gait vs Clinical Weight")
            if alpha_column:
                fusion_plot=pd.DataFrame({"Patient":patient_labels,"Gait Weight (α)":pd.to_numeric(test_predictions[alpha_column],errors="coerce"),"Predicted Severity":predicted.astype(str)}).dropna(subset=["Gait Weight (α)"])
                if plotly_available() and not fusion_plot.empty:
                    fig=px.bar(fusion_plot,x="Patient",y="Gait Weight (α)",color="Predicted Severity",text=fusion_plot["Gait Weight (α)"].map(lambda x:f"{x:.2f}"),
                               category_orders={"Predicted Severity":CLASS_NAMES},color_discrete_sequence=["#06b6d4","#8b5cf6","#f59e0b"],title="Patient-Specific Adaptive Fusion Weight")
                    fig.update_yaxes(range=[0,1],title="Gait Contribution Weight (α)"); fig.update_layout(height=420,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)")
                    st.plotly_chart(fig,use_container_width=True,key="fusion_bar_graph")
                    st.caption("α represents the learned gait-side fusion weight; clinical-side weight is 1−α. It is a model fusion coefficient, not a clinical importance score.")
                else:
                    st.dataframe(fusion_plot,use_container_width=True,hide_index=True)
            else:
                st.info("Adaptive Alpha is not present in the saved prediction output.")

            # ----------------------------------------------------
            # 6. Gait signal / temporal explanation
            # ----------------------------------------------------
            st.markdown("### 6. Patient Gait Signal & Temporal Explanation")
            patient_options=patient_ids_from_manifest()
            if patient_options:
                gp=st.selectbox("Select CMT1A patient",patient_options,key="evaluation_gait_patient")
                render_patient_gait(gp,"evaluation_gait")
                temporal_path=os.path.join(RESULTS_DIR,f"{gp}_gait_temporal_explanation.csv")
                if os.path.exists(temporal_path):
                    tdf=load_csv(temporal_path)
                    if tdf is not None and not tdf.empty and plotly_available():
                        numeric=tdf.select_dtypes(include=np.number)
                        cols=[c for c in numeric.columns if c.lower() not in {"sample","window","index"}][:2]
                        xcol="Sample" if "Sample" in tdf.columns else tdf.columns[0]
                        if cols:
                            fig=px.line(tdf,x=xcol,y=cols,title=f"Temporal Gait Explanation — {gp}",color_discrete_sequence=["#06b6d4","#8b5cf6"])
                            fig.update_layout(height=410,xaxis_title="Sample / Time",yaxis_title="Signal / Attribution",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)")
                            st.plotly_chart(fig,use_container_width=True,key="temporal_eval_graph")
            else:
                st.info("No CMT1A patient manifest is available for gait visualization.")

            # ----------------------------------------------------
            # 10. Reviewer interpretation
            # ----------------------------------------------------
            st.markdown("### 10. Evaluation Interpretation")
            st.info("These graphs summarize the currently saved patient-level evaluation output. The displayed metrics should be interpreted together with the very small held-out test cohort and its class composition.")


#  ADAPTIVE FUSION


elif page == "Adaptive Fusion":

    st.header(
        "Adaptive Clinical Fusion"
    )

    st.write(
        "The fusion layer learns how strongly the gait and "
        "clinical representations contribute to each patient's "
        "final multimodal representation."
    )


    if test_predictions is None:

        st.error(
            "Test prediction data are unavailable."
        )

    else:

        patient_column = find_column(
            test_predictions,
            [
                "Patient_ID",
                "ID",
                "Patient"
            ]
        )

        alpha_column = find_column(
            test_predictions,
            [
                "Adaptive_Alpha",
                "Alpha"
            ]
        )

        if alpha_column is None:

            st.error(
                "Adaptive_Alpha was not found."
            )

        else:

            fusion_table = pd.DataFrame()

            if patient_column:

                fusion_table[
                    "Patient"
                ] = test_predictions[
                    patient_column
                ]

            fusion_table[
                "Adaptive Alpha"
            ] = test_predictions[
                alpha_column
            ]


            st.dataframe(
                fusion_table,
                use_container_width=True,
                hide_index=True
            )


            if patient_column:

                patient = st.selectbox(
                    "Select patient",
                    fusion_table[
                        "Patient"
                    ].astype(str)
                )

                selected = fusion_table[
                    fusion_table[
                        "Patient"
                    ].astype(str)
                    == patient
                ]

                alpha = float(
                    selected[
                        "Adaptive Alpha"
                    ].iloc[0]
                )


                st.markdown(
                    "### Patient-Specific Fusion"
                )


                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "Gait Representation Weight",
                        f"{alpha:.4f}"
                    )

                with c2:

                    st.metric(
                        "Clinical Representation Weight",
                        f"{1-alpha:.4f}"
                    )


                st.progress(
                    min(
                        max(alpha, 0.0),
                        1.0
                    )
                )


                st.caption(
                    "Adaptive Alpha is a learned fusion "
                    "coefficient. It describes model fusion "
                    "behavior rather than clinical importance."
                )


    # --------------------------------------------------------
    # Stage 9 explanation files
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Adaptive Fusion Explanation"
    )

    fusion_files = sorted(
        glob.glob(
            os.path.join(
                RESULTS_DIR,
                "*_adaptive_fusion.csv"
            )
        )
    )


    if fusion_files:

        for path in fusion_files:

            patient_name = (
                os.path.basename(path)
                .replace(
                    "_adaptive_fusion.csv",
                    ""
                )
            )

            df = load_csv(path)

            if df is not None:

                st.markdown(
                    f"**{patient_name}**"
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

    else:

        st.info(
            "No patient-specific adaptive-fusion "
            "explanation file was found."
        )


# ============================================================
# PAGE 5 — DUAL AXIS
# ============================================================

elif page == "Dual-Axis Analysis":

    st.header(
        "Dual-Axis Representation"
    )

    st.write(
        "The model learns complementary representations "
        "associated with the joint multimodal space, "
        "severity representation and impairment representation."
    )

    if dual_axis_df is None:

        st.warning(
            "Dual-axis result file is unavailable."
        )

    else:

        st.dataframe(
            dual_axis_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 6 — EXPLAINABLE AI
# ============================================================

elif page == "Explainable AI":
    st.markdown("## Explainable AI")
    st.caption("Two complementary patient-level views: clinical feature contribution and temporal gait analysis.")
    patients=patient_ids_from_manifest()
    if not patients:
        st.error("No CMT1A patients found.")
    else:
        patient=st.selectbox("Select CMT1A patient",patients,key="xai_patient")
        prow=patient_row_with_clinical(patient)
        info=gait_window_summary(patient)
        a,b,c,d=st.columns(4)
        a.metric("Patient",patient)
        b.metric("Age",clinical_value(prow,["Age (Years)","Age","AgeYears"]) or "N/A")
        c.metric("Severity",str(prow.iloc[0].get("severity_name","N/A")))
        d.metric("Gait Windows",info["windows"])

        st.markdown("### 1. Clinical Feature Contribution")
        shap_path=os.path.join(RESULTS_DIR,f"{patient}_clinical_shap.csv")
        if os.path.exists(shap_path):
            sdf=load_csv(shap_path)
            if sdf is not None and not sdf.empty:
                st.dataframe(sdf,use_container_width=True,hide_index=True)
                num=sdf.select_dtypes(include=np.number)
                if not num.empty and plotly_available():
                    score_col=next((c for c in ["SHAP Value","shap_value","Contribution","mean_abs_shap"] if c in sdf.columns),num.columns[-1])
                    feat_col=next((c for c in ["Feature","feature","Feature Name"] if c in sdf.columns),sdf.columns[0])
                    plot=sdf[[feat_col,score_col]].copy(); plot[score_col]=pd.to_numeric(plot[score_col],errors="coerce"); plot=plot.dropna().sort_values(score_col)
                    fig=px.bar(plot,x=score_col,y=feat_col,orientation="h",title=f"Clinical Feature Contribution — {patient}")
                    fig.update_layout(height=400,xaxis_title="Contribution",yaxis_title="Clinical Feature")
                    st.plotly_chart(fig,use_container_width=True)
        else:
            fallback=build_clinical_fallback(patient)
            if fallback.empty:
                st.info("Clinical contribution file not generated for this patient yet.")
            else:
                st.caption("Cohort-normalized clinical sensitivity view — not labelled as SHAP.")
                st.dataframe(fallback,use_container_width=True,hide_index=True)
                if plotly_available():
                    fig=px.bar(fallback.sort_values("Relative Contribution"),x="Relative Contribution",y="Feature",orientation="h",title=f"Clinical Feature Sensitivity — {patient}")
                    fig.update_layout(height=390)
                    st.plotly_chart(fig,use_container_width=True)

        st.markdown("### 2. Temporal Gait Analysis")
        temporal_path=os.path.join(RESULTS_DIR,f"{patient}_gait_temporal_explanation.csv")
        if os.path.exists(temporal_path):
            tdf=load_csv(temporal_path)
        else:
            tdf,_=build_temporal_fallback(patient)
            if not tdf.empty: st.caption("Patient-specific temporal signal sensitivity view — not labelled as a saved attribution file.")
        if tdf is None or tdf.empty:
            st.info("No patient-linked gait data are available for temporal analysis.")
        else:
            st.dataframe(tdf.head(256),use_container_width=True,hide_index=True)
            if plotly_available():
                numeric=tdf.select_dtypes(include=np.number)
                cols=[c for c in numeric.columns if c.lower() not in {"sample","window","index"}]
                if cols:
                    xcol="Sample" if "Sample" in tdf.columns else tdf.columns[0]
                    fig=px.line(tdf,x=xcol,y=cols[:2],title=f"Temporal Gait Analysis — {patient}")
                    fig.update_layout(height=420,xaxis_title="Sample / Time",yaxis_title="Signal / Attribution")
                    st.plotly_chart(fig,use_container_width=True)
            render_patient_gait(patient, "xai_gait")


#  RESEARCH & EVIDENCE


elif page == "Research & Evidence":

    st.header("Research & Evidence")
    st.write("Patient-specific literature retrieval and evidence-grounded clinical interpretation for CMT1A.")

    # --------------------------------------------------------
    # PATIENT-SPECIFIC RETRIEVAL
    # --------------------------------------------------------
    st.subheader("Patient-Specific Evidence Retrieval")

    if manifest is not None:
        id_col = find_column(manifest, ["ID", "Patient_ID", "Patient"])
        patient_options = sorted(manifest[id_col].dropna().astype(str).unique().tolist()) if id_col else []
    else:
        patient_options = []

    if not patient_options:
        patient_options = explanation_patients

    if not patient_options:
        st.warning("No CMT1A patients are available for evidence retrieval.")
    else:
        selected_evidence_patient = st.selectbox(
            "Select CMT1A patient",
            patient_options,
            key="evidence_patient"
        )

        # Show the actual patient context used to construct the retrieval query.
        context_cols = st.columns(4)
        prow = manifest[manifest[id_col].astype(str) == str(selected_evidence_patient)] if manifest is not None and id_col else pd.DataFrame()
        if not prow.empty:
            r = prow.iloc[0]
            with context_cols[0]:
                st.metric("Patient", selected_evidence_patient)
            with context_cols[1]:
                st.metric("Severity", str(r.get("severity_name", "N/A")).title())
            with context_cols[2]:
                st.metric("CMTES", str(r.get("Cmtes_score", "N/A")))
            with context_cols[3]:
                st.metric("CMT-FOM", str(r.get("CMT-FOM_Total", "N/A")))

        patient_query = build_patient_evidence_query(selected_evidence_patient)
        patient_query = st.text_area(
            "Patient-specific retrieval query",
            value=patient_query,
            height=85,
            key="patient_evidence_query"
        )

        top_k = st.slider("Number of retrieved evidence passages", 2, 10, 5, key="patient_top_k")

        if st.button("Retrieve Patient Evidence", type="primary"):
            if rag_engine is None:
                st.error("The local CMT evidence engine could not be loaded.")
            else:
                try:
                    results = rag_engine.retrieve_evidence(patient_query, top_k=top_k)
                    if not results:
                        st.warning("No relevant CMT1A evidence was retrieved for this patient.")
                    else:
                        # Store so the evidence remains visible after Streamlit reruns.
                        st.session_state["patient_evidence_results"] = results
                        st.session_state["patient_evidence_patient"] = selected_evidence_patient
                except Exception as e:
                    st.error(f"Patient evidence retrieval failed: {e}")

        stored_results = st.session_state.get("patient_evidence_results", [])
        stored_patient = st.session_state.get("patient_evidence_patient")
        if stored_results and stored_patient == selected_evidence_patient:
            st.markdown("#### Retrieved Evidence")

            # Reviewer-friendly table: concise finding rather than raw chunks.
            rows = []
            for i, item in enumerate(stored_results[:top_k], start=1):
                rows.append({
                    "Evidence": f"Evidence {i}",
                    "Source": str(item.get("source", "Unknown source")),
                    "Similarity": float(item.get("score", 0)) if str(item.get("score", "")).strip() else 0.0,
                    "Important Evidence": extract_important_evidence(item.get("text", ""), max_sentences=2, max_chars=560)
                })
            evidence_df = pd.DataFrame(rows)
            st.dataframe(
                evidence_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Similarity": st.column_config.NumberColumn(format="%.4f"),
                    "Important Evidence": st.column_config.TextColumn(width="large")
                }
            )

            st.markdown("#### Evidence Details")
            render_evidence_cards(stored_results[:top_k], selected_evidence_patient)

    # --------------------------------------------------------
    # GENERAL LITERATURE RETRIEVAL
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("General CMT1A Literature Retrieval")
    default_query = (
        "CMT1A disease severity gait abnormalities quantitative gait characteristics "
        "CMT-FOM CMT neuropathy score clinical assessment muscle weakness foot drop balance "
        "functional impairment longitudinal progression"
    )
    query = st.text_area("Research question", value=default_query, height=90, key="general_query")
    general_top_k = st.slider("Evidence passages", 1, 10, 5, key="general_top_k")

    if st.button("Retrieve General Literature"):
        if rag_engine is None:
            st.error("The local CMT evidence engine could not be loaded.")
        else:
            try:
                results = rag_engine.retrieve_evidence(query, top_k=general_top_k)
                if not results:
                    st.warning("No relevant literature was found.")
                else:
                    st.session_state["general_evidence_results"] = results
            except Exception as e:
                st.error(f"Literature retrieval failed: {e}")

    general_results = st.session_state.get("general_evidence_results", [])
    if general_results:
        render_evidence_cards(general_results[:general_top_k])

    # --------------------------------------------------------
    # LITERATURE REFERENCES TO SUPPORT GAIT ANALYSIS
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("CMT1A Literature Supporting Gait Analysis")
    st.caption("These studies are used as literature evidence and interpretation references. Their patients are not added to your training labels unless compatible raw data are separately imported and validated.")
    literature_refs = pd.DataFrame([
        {"Paper": "Gait Pattern in CMT1A According to Disease Severity (2023)", "Cohort": "22 CMT1A + 22 controls", "Useful findings": "Gait speed, step length, hip/knee/ankle kinematics and kinetics", "Role": "Gait-severity reference"},
        {"Paper": "Gait pattern classification in children with CMT1A (2014)", "Cohort": "21 CMT1A + 18 controls", "Useful findings": "Gait pattern, foot-drop and kinematic features", "Role": "Gait-pattern reference"},
        {"Paper": "Variation in SIPA1L2 is correlated with phenotype modification in CMT1A (2020)", "Cohort": "971 ascertained CMT1A patients", "Useful findings": "CMT1A phenotype and dorsiflexion weakness", "Role": "Large-cohort clinical context"}
    ])
    st.dataframe(literature_refs, use_container_width=True, hide_index=True)
    st.caption("The 971-patient study provides large clinical phenotype evidence; it is not 971 gait recordings. Only compatible raw gait data should be used for gait-model training.")

    # --------------------------------------------------------
    # SAVED PATIENT EXPLANATION
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("Patient Evidence")

    # IMPORTANT: use all CMT1A patients, not only patients with saved LLM files.
    all_patient_options = patient_options
    selected_patient = st.selectbox(
        "Patient explanation",
        all_patient_options,
        key="saved_patient_explanation"
    )

    matching_llm = os.path.join(RESULTS_DIR, f"{selected_patient}_llm_explanation.txt")
    if not os.path.exists(matching_llm):
        st.info("No saved LLM explanation exists for this patient yet. Patient-specific literature retrieval above can still be used.")
    else:
        try:
            with open(matching_llm, "r", encoding="utf-8") as file:
                explanation = clean_llm_text(file.read())

            sections = [
                "1. Model Prediction",
                "2. Model Explanation",
                "3. Relevant Literature Evidence",
                "4. Evidence-Based Clinical Support",
                "5. Limitations"
            ]
            parsed = {}
            current = None
            for line in explanation.splitlines():
                stripped = line.strip()
                clean_line = stripped.replace("**", "").strip()
                matched = False
                for section in sections:
                    if clean_line.startswith(section):
                        current = section
                        parsed[current] = []
                        matched = True
                        break
                if not matched and current and stripped:
                    parsed[current].append(stripped)

            # Render the literature evidence as individual readable evidence blocks.
            for section in sections:
                if section not in parsed:
                    continue
                title = re.sub(r"^\d+\.\s*", "", section)
                content_lines = [x for x in parsed[section] if x]
                if not content_lines:
                    continue
                st.markdown(f"### {title}")
                if section == "3. Relevant Literature Evidence":
                    for idx, line in enumerate(content_lines, start=1):
                        line = re.sub(r"^[\-*•]+\s*", "", line)
                        st.markdown(
                            f"<div class=\"evidence-card\"><b>Evidence {idx}</b><br><span style='line-height:1.6'>{line}</span></div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown("\n".join(content_lines))
                st.markdown("---")
        except Exception as e:
            st.error(f"Could not read explanation: {e}")

    
    # PIPELINE
    
    st.subheader("Evidence Pipeline")
    st.write("CMT1A patient context → targeted retrieval → ranked literature → important finding extraction → patient explanation")




elif page == "Model Results":

    st.header(
        "Model Results"
    )


    
    st.subheader(
        "Model Architecture"
    )

    st.write(
        """
        **Gait branch**

        InceptionTime → Temporal Transformer → Window Attention


        **Clinical branch**

        Clinical MLP


        **Adaptive fusion**

        Patient-specific gait–clinical representation fusion


        **Representation**

        Dual-Axis Representation


        **Output**

        Mild / Moderate / Severe
        """
    )


    
    # Pipeline status
    

    st.subheader(
        "Pipeline Status"
    )

    status = pd.DataFrame(
        {
            "Component": [
                "Patient preparation",
                "Multimodal architecture",
                "Model checkpoint",
                "Held-out evaluation",
                "Explainability",
                "Dual-axis analysis",
                "Literature retrieval",
                "Evidence-grounded explanation"
            ],

            "Status": [
                "Available"
                if manifest is not None
                else "Missing",

                "Implemented",

                "Available"
                if CHECKPOINT_PATH
                else "Missing",

                "Available"
                if test_predictions is not None
                else "Missing",

                "Available"
                if explanation_patients
                else "Missing",

                "Available"
                if dual_axis_df is not None
                else "Missing",

                "Available"
                if rag_engine
                else "Missing",

                "Available"
                if llm_files
                else "Missing"
            ]
        }
    )

    st.dataframe(
        status,
        use_container_width=True,
        hide_index=True
    )


    

    st.subheader(
        "Model Configuration"
    )

    config = pd.DataFrame(
        {
            "Component": [
                "Gait encoder",
                "Temporal model",
                "Window attention",
                "Clinical encoder",
                "Fusion",
                "Representation",
                "Classes",
                "Parameters"
            ],

            "Configuration": [
                "InceptionTime",
                "Temporal Transformer",
                "Learned Window Attention",
                "Clinical MLP",
                "Adaptive Clinical Fusion",
                "Dual-Axis Representation",
                "3",
                "313,013"
            ]
        }
    )

    st.dataframe(
        config,
        use_container_width=True,
        hide_index=True
    )


    

    st.subheader(
        "Research Data Summary"
    )

    data_summary = pd.DataFrame(
        {
            "Data Component": [
                "CMT1A clinical-gait data",
                "Multi-sensor gait data",
                "Additional gait data",
                "Combined supplementary gait"
            ],

            "Scale": [
                f"{len(manifest) if manifest is not None else 0} subjects",

                f"{gait_summary['multi_users']} users · "
                f"{gait_summary['multi_recordings']} recordings · "
                f"{gait_summary['multi_windows']} windows",

                f"{gait_summary['secondary_windows']} windows",

                f"{gait_summary['combined_windows']} windows"
            ]
        }
    )

    st.dataframe(
        data_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    CMT-SCOPE AI · Explainable AI-Based CMT Disease
    Severity Prediction System · Research Prototype
    </div>
    """,
    unsafe_allow_html=True
)