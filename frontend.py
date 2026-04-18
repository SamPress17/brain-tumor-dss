"""
Brain Tumor DSS — Medical Grade Streamlit Frontend
Run: python -m streamlit run frontend.py
"""

import streamlit as st
import requests, base64, io, json
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor DSS",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:5000"
AUTH_TOKEN_KEY = "auth_token"

if "auth_token" not in st.session_state:
    st.session_state["auth_token"] = ""
if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = ""
if "auth_error" not in st.session_state:
    st.session_state["auth_error"] = ""
if "report_pdf" not in st.session_state:
    st.session_state["report_pdf"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []


def get_auth_headers():
    token = st.session_state.get("auth_token", "")
    return {"X-Api-Key": token} if token else {}


def authenticate(username, password):
    try:
        resp = requests.post(
            f"{API_URL}/login",
            json={"username": username, "password": password},
            timeout=6,
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state["auth_token"] = data.get("token", "")
            st.session_state["auth_user"] = data.get("user", username)
            st.session_state["auth_error"] = ""
            return True
        st.session_state["auth_error"] = resp.json().get("error", "Invalid credentials")
    except Exception as exc:
        st.session_state["auth_error"] = f"Auth failed: {exc}"
    return False


API_URL = "http://localhost:5000"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;font-size:13px}
.stApp{background:#f0f4f8}
.block-container{padding:1rem 1rem 2rem!important;max-width:780px!important}
#MainMenu,footer,header{visibility:hidden}
div[data-testid="stToolbar"]{display:none}

/* Top bar */
.topbar{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:10px 16px;display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.sys-name{font-size:14px;font-weight:600;color:#0f172a}
.sys-sub{font-size:11px;color:#64748b}
.logo-badge{background:#0C447C;color:#B5D4F4;border-radius:6px;padding:4px 10px;font-size:11px;font-weight:600;letter-spacing:.04em}
.status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#16a34a;margin-right:5px}

/* Patient bar */
.patient-bar{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:10px 16px;display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px}
.pfield{font-size:10px;color:#64748b}
.pfield b{display:block;font-size:12px;font-weight:500;color:#0f172a;margin-top:1px}

/* Cards */
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;margin-bottom:10px}
.card-title{font-size:10px;font-weight:600;color:#64748b;letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px}
.card-tag{font-size:10px;padding:2px 8px;border-radius:999px;font-weight:500}
.tag-mri{background:#E6F1FB;color:#185FA5;border:1px solid #B5D4F4}
.tag-ai{background:#EEEDFE;color:#534AB7;border:1px solid #AFA9EC}

/* Image label */
.img-label{text-align:center;font-size:10px;color:#94a3b8;font-weight:500;text-transform:uppercase;letter-spacing:.06em;margin-top:4px;padding:5px;background:#f8fafc;border-top:1px solid #e2e8f0;border-radius:0 0 10px 10px}

/* Diagnosis */
.dx-name{font-size:26px;font-weight:600;color:#0f172a;letter-spacing:-.02em;line-height:1}
.dx-conf{font-family:'JetBrains Mono',monospace;font-size:11px;color:#64748b;margin-top:3px}
.sev-critical{background:#fef2f2;color:#b91c1c;border:1px solid #fecaca;border-radius:6px;padding:8px 12px;text-align:center}
.sev-moderate{background:#fffbeb;color:#b45309;border:1px solid #fde68a;border-radius:6px;padding:8px 12px;text-align:center}
.sev-normal{background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;border-radius:6px;padding:8px 12px;text-align:center}
.sev-word{font-size:13px;font-weight:600}
.sev-urgency{font-size:10px;margin-top:2px;font-weight:500;letter-spacing:.04em}

/* Meter */
.meter{display:flex;gap:3px;margin:8px 0 4px}
.mdot{height:5px;border-radius:3px;flex:1}
.mon-red{background:#ef4444}
.mon-yel{background:#f59e0b}
.mon-grn{background:#22c55e}
.moff{background:#e2e8f0}

/* Info boxes */
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}
.info-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;padding:10px 12px}
.info-label{font-size:10px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.info-text{font-size:12px;color:#334155;line-height:1.5}
.info-text b{font-weight:600;color:#b91c1c}

/* Prob row */
.prob-row{display:flex;align-items:center;gap:10px;margin-bottom:7px}
.prob-row:last-child{margin-bottom:0}
.pname{font-size:12px;color:#334155;width:82px;flex-shrink:0}
.pbg{flex:1;height:7px;background:#f1f5f9;border-radius:4px;overflow:hidden}
.pfill{height:100%;border-radius:4px}
.ppct{font-family:'JetBrains Mono',monospace;font-size:11px;color:#64748b;width:38px;text-align:right;flex-shrink:0}
.ptop{font-size:10px;font-weight:600;padding:1px 7px;border-radius:999px;background:#fef2f2;color:#b91c1c;border:1px solid #fecaca;flex-shrink:0}

/* Stats */
.stat-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.stat-box{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px}
.stat-label{font-size:10px;color:#64748b;margin-bottom:3px}
.stat-val{font-size:20px;font-weight:600;color:#0f172a;font-family:'JetBrains Mono',monospace}
.stat-sub{font-size:10px;color:#94a3b8;margin-top:2px}

/* Disclaimer */
.disclaimer{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;font-size:11px;color:#854f0b;line-height:1.6}
</style>
""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────

def b64img(b64): return Image.open(io.BytesIO(base64.b64decode(b64)))

def meter_html(level):
    scores = {"Critical":5,"Moderate":3,"Normal":1}
    cls    = {"Critical":"mon-red","Moderate":"mon-yel","Normal":"mon-grn"}
    n = scores.get(level,1); c = cls.get(level,"mon-grn")
    dots = "".join(f'<div class="mdot {c if i<=n else "moff"}"></div>' for i in range(1,6))
    return f'<div class="meter">{dots}</div><div style="font-size:10px;color:#94a3b8">Severity: {n} / 5</div>'

def sev_html(level,urgency):
    cls = {"Critical":"sev-critical","Moderate":"sev-moderate","Normal":"sev-normal"}.get(level,"sev-normal")
    return f'<div class="{cls}"><div class="sev-word">{level}</div><div class="sev-urgency">{urgency}</div></div>'


# ── top bar ───────────────────────────────────────────────

try:
    r = requests.get(f"{API_URL}/health", timeout=2)
    api_ok = r.status_code == 200
except:
    api_ok = False

status = '<span class="status-dot"></span>API online' if api_ok else '⚠ API offline — run: python -m flask --app app run'
auth_label = f"Signed in as {st.session_state.get('auth_user')}" if st.session_state.get("auth_token") else "Authentication required"
st.markdown(f"""
<div class="topbar">
  <div>
    <div class="sys-name">Brain Tumor Decision Support System</div>
    <div class="sys-sub">{auth_label}</div>
  </div>
  <div style="display:flex;align-items:center;gap:10px">
    <span style="font-size:11px;color:{'#16a34a' if api_ok else '#dc2626'}">{status}</span>
    <span class="logo-badge">DSS</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── patient bar ───────────────────────────────────────────

if not st.session_state.get("auth_token"):
    with st.expander("Clinician login", expanded=True):
        st.info("Sign in to access scan upload, report generation, and history.")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign in", key="login_submit"):
            authenticate(login_user, login_pass)
        if st.session_state.get("auth_error"):
            st.error(st.session_state.get("auth_error"))
    st.stop()

with st.expander("Patient details (optional)", expanded=False):
    c1,c2,c3,c4 = st.columns(4)
    pid  = c1.text_input("Patient ID",  value="PT-00001",  label_visibility="visible")
    age  = c2.text_input("Age / Sex",   value="—",         label_visibility="visible")
    scan = c3.text_input("Scan type",   value="T1 MRI",    label_visibility="visible")
    ref  = c4.text_input("Referred by", value="—",         label_visibility="visible")

st.markdown(f"""
<div class="patient-bar">
  <div style="display:flex;gap:20px;flex-wrap:wrap">
    <div class="pfield">Patient ID<b>{pid}</b></div>
    <div class="pfield">Age / Sex<b>{age}</b></div>
    <div class="pfield">Scan type<b>{scan}</b></div>
    <div class="pfield">Referred by<b>{ref}</b></div>
  </div>
</div>
""", unsafe_allow_html=True)

if st.button("Logout", key="logout"): 
    st.session_state["auth_token"] = ""
    st.session_state["auth_user"] = ""
    st.session_state["auth_error"] = ""
    st.session_state["report_pdf"] = None
    st.session_state["history"] = []
    st.experimental_rerun()


# ── upload ────────────────────────────────────────────────

uploaded = st.file_uploader("Upload MRI scan (JPG / PNG)", type=["jpg","jpeg","png"])

if not uploaded:
    st.info("Upload an MRI scan above to begin analysis.")
    st.stop()


# ── api call ──────────────────────────────────────────────

with st.spinner("Analysing scan..."):
    uploaded.seek(0)
    metadata = json.dumps({
        "patient": {
            "patient_id": pid,
            "age_sex": age,
            "scan_type": scan,
            "referred_by": ref,
        }
    })
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            files={"file":(uploaded.name, uploaded.read(), uploaded.type)},
            data={"metadata": metadata},
            headers=get_auth_headers(),
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach Flask API. Run: python -m flask --app app run --host=0.0.0.0 --port=5000")
        st.stop()
    except Exception as e:
        st.error(f"Error: {e}"); st.stop()

pred    = data["prediction"].capitalize()
conf    = data["confidence"]
probs   = data["probabilities"]
sev     = data["severity"]
cam_b64 = data["gradcam"]
ori_b64 = data["original"]
level   = sev["level"]
urgency = sev["urgency"]
desc    = sev["description"]
action  = sev["action"]


# ── scan images ───────────────────────────────────────────

ci, cg = st.columns(2, gap="small")
with ci:
    st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px"><span style="font-size:10px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.06em">Original MRI scan</span><span class="card-tag tag-mri" style="font-size:10px;padding:2px 8px;border-radius:999px;font-weight:500;background:#E6F1FB;color:#185FA5;border:1px solid #B5D4F4">Axial T1</span></div>', unsafe_allow_html=True)
    st.image(b64img(ori_b64), use_column_width=True)
    st.markdown('<div class="img-label">Slice · uploaded scan</div>', unsafe_allow_html=True)

with cg:
    st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px"><span style="font-size:10px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.06em">Grad-CAM activation</span><span class="card-tag tag-ai" style="font-size:10px;padding:2px 8px;border-radius:999px;font-weight:500;background:#EEEDFE;color:#534AB7;border:1px solid #AFA9EC">AI attention</span></div>', unsafe_allow_html=True)
    st.image(b64img(cam_b64), use_column_width=True)
    st.markdown('<div class="img-label">Red = highest activation region</div>', unsafe_allow_html=True)


# ── diagnosis result ──────────────────────────────────────

cr, cs = st.columns([2,1], gap="small")
with cr:
    st.markdown(f"""
    <div class="card" style="margin-top:10px">
      <div class="card-title">Primary diagnosis</div>
      <div class="dx-name">{pred}</div>
      <div class="dx-conf">Confidence: {conf}% &nbsp;·&nbsp; Model: ResNet-50</div>
      {meter_html(level)}
      <div class="two-col">
        <div class="info-box">
          <div class="info-label">Clinical description</div>
          <div class="info-text">{desc}</div>
        </div>
        <div class="info-box">
          <div class="info-label">Recommended action</div>
          <div class="info-text"><b>{action}</b></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with cs:
    st.markdown(f"""
    <div class="card" style="margin-top:10px;text-align:center">
      <div class="card-title">Severity level</div>
      {sev_html(level, urgency)}
    </div>
    """, unsafe_allow_html=True)


# ── probability breakdown ─────────────────────────────────

CLASS_COLORS = {"glioma":"#E24B4A","meningioma":"#BA7517","notumor":"#3B6D11","pituitary":"#185FA5"}
sorted_p = sorted(probs.items(), key=lambda x:x[1], reverse=True)

st.markdown("""
<div class="card">
  <div class="card-title">Differential probability breakdown</div>
</div>
""", unsafe_allow_html=True)

for i,(cls,pct) in enumerate(sorted_p):
    top_html = '<span class="ptop">top</span>' if i == 0 else ""
    c1, c2, c3 = st.columns([2, 10, 1], gap="small")
    c1.markdown(f"<span style='font-size:12px;color:#334155'>{cls.capitalize()}</span>", unsafe_allow_html=True)
    c2.progress(pct / 100)
    c3.markdown(
        f"<div style='text-align:right;font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#64748b'>{pct}% {top_html}</div>",
        unsafe_allow_html=True,
    )

report_payload = {
    "patient": {"patient_id": pid, "age_sex": age, "scan_type": scan, "referred_by": ref},
    "prediction": pred,
    "confidence": conf,
    "probabilities": probs,
    "severity": sev,
    "original_b64": ori_b64,
    "gradcam_b64": cam_b64,
    "segmentation_overlay_b64": data.get("segmentation_overlay"),
    "segmentation_mask_b64": data.get("segmentation_mask"),
    "segmentation_area_pct": data.get("segmentation_area_pct", 0),
    "timestamp": data.get("timestamp", ""),
}

if st.button("Generate PDF report", key="generate_pdf"):
    try:
        resp = requests.post(
            f"{API_URL}/report",
            json=report_payload,
            headers=get_auth_headers(),
            timeout=20,
        )
        resp.raise_for_status()
        st.session_state["report_pdf"] = resp.content
        st.success("PDF report is ready.")
    except Exception as err:
        st.error(f"Report generation failed: {err}")

if st.session_state.get("report_pdf"):
    st.download_button(
        "Download PDF report",
        st.session_state["report_pdf"],
        file_name=f"brain_tumor_report_{pid or 'scan'}.pdf",
        mime="application/pdf",
    )

st.markdown("""
<div class="card">
  <div class="card-title">Tumor segmentation</div>
</div>
""", unsafe_allow_html=True)
seg1, seg2 = st.columns(2, gap="small")
with seg1:
    st.image(b64img(data["segmentation_overlay"]), caption="Segmentation overlay", use_column_width=True)
with seg2:
    st.image(b64img(data["segmentation_mask"]), caption="Segmentation mask", use_column_width=True)
st.markdown(f"<div style='font-size:11px;color:#64748b;margin-top:10px'>Estimated tumor coverage: {data.get('segmentation_area_pct', 0)}%</div>", unsafe_allow_html=True)

if st.button("Refresh scan history", key="refresh_history"):
    try:
        history_resp = requests.get(f"{API_URL}/history", headers=get_auth_headers(), timeout=10)
        history_resp.raise_for_status()
        st.session_state["history"] = history_resp.json().get("history", [])
    except Exception as err:
        st.error(f"Cannot load history: {err}")

if st.session_state.get("history"):
    st.markdown("""
<div class="card">
  <div class="card-title">Scan history / patient log</div>
</div>
""", unsafe_allow_html=True)
    history_rows = []
    for item in st.session_state["history"][:12]:
        history_rows.append({
            "Time": item.get("timestamp", ""),
            "Patient": item.get("patient", {}).get("patient_id", "—"),
            "Scan": item.get("patient", {}).get("scan_type", "—"),
            "Result": item.get("prediction", "").capitalize(),
            "Confidence": f"{item.get('confidence', 0)}%",
            "Tumor area": f"{item.get('segmentation_area_pct', 0)}%",
        })
    st.table(history_rows)
else:
    st.info("Scan history is empty. Perform a scan to populate the patient log.")


# ── confidence gauge ──────────────────────────────────────

gc = {"Critical":"#E24B4A","Moderate":"#BA7517","Normal":"#3B6D11"}.get(level,"#185FA5")
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=conf,
    number={"suffix":"%","font":{"size":26,"color":"#0f172a","family":"JetBrains Mono"}},
    gauge={
        "axis":{"range":[0,100],"tickfont":{"size":10,"color":"#94a3b8"},"tickcolor":"#e2e8f0"},
        "bar":{"color":gc,"thickness":0.22},
        "bgcolor":"#f8fafc","bordercolor":"#e2e8f0",
        "steps":[{"range":[0,40],"color":"#f8fafc"},{"range":[40,70],"color":"#f1f5f9"},{"range":[70,100],"color":"#eef2ff"}],
        "threshold":{"line":{"color":gc,"width":2},"thickness":0.8,"value":conf},
    },
    title={"text":"Confidence score","font":{"size":11,"color":"#64748b"}},
))
fig.update_layout(paper_bgcolor="white",font=dict(family="Inter"),
                  height=170,margin=dict(l=16,r=16,t=28,b=8))
st.plotly_chart(fig, use_container_width=True)


# ── model stats ───────────────────────────────────────────

st.markdown("""
<div class="stat-row">
  <div class="stat-box">
    <div class="stat-label">Model accuracy (test set)</div>
    <div class="stat-val">96.2%</div>
    <div class="stat-sub">ResNet-50 · 1,311 test images · 4 classes</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Macro ROC-AUC</div>
    <div class="stat-val">0.993</div>
    <div class="stat-sub">One-vs-rest · Glioma / Meningioma / Pituitary / None</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── disclaimer ────────────────────────────────────────────

st.markdown("""
<div class="disclaimer">
  <b>Clinical disclaimer:</b> This system is designed to assist qualified medical professionals
  and is not a substitute for clinical judgment. All findings must be verified by a licensed
  radiologist or neurosurgeon before clinical decisions are made.
  For research and educational purposes only.
</div>
""", unsafe_allow_html=True)