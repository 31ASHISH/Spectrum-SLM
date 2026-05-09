"""
app_phase2.py — Spectrum-SLM Phase 2 Dashboard
================================================
Streamlit app for the NEW dataset model (5-class: BPSK/QPSK/8PSK/16QAM/DQPSK).

Differences from app.py:
  - n_mod_classes = 5 (adds DQPSK)
  - Loads checkpoint from checkpoints/phase2/slm_phase2_new_best.pt
  - Loads normalizer from checkpoints/phase2/normalizer_phase2.pkl
  - Shows metrics_phase2.json in Research tab
  - Dataset Explorer tab for browsing new dataset samples

Run: streamlit run app_phase2.py

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P. | IIT Palakkad
Dated   : April 2026
"""

import os, sys, pickle, time, json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.graph_objects as go                                                                                                                                                                                                                                                                                                                                               
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from spectrum_slm_model import SpectrumSLM
from spectrum_slm_dataset import N_BINS
from config import (
    CKPT_PHASE2, CKPT_PHASE3, CKPT_PHASE2_BEST, NORMALIZER_FILE, METRICS_FILE,
    MOD_NAMES_V2, MOD_COLORS_V2, N_MOD_CLASSES_V2, MOD_MAP_V2,
    NEW_DATASET_DIR, SECONDARY_USER_DIR,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spectrum-SLM Phase 2 | New Dataset",
    page_icon="📡", layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html,body,[class*="css"]{font-family:'Inter',sans-serif;}
  .main{background:#0d1117;}
  .metric-card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:16px;text-align:center;}
  .metric-value{font-size:2em;font-weight:700;color:#58a6ff;}
  .metric-label{font-size:0.8em;color:#8b949e;margin-top:4px;}
  .phase2-badge{display:inline-block;background:linear-gradient(135deg,#533483,#0f3460);
    border-radius:20px;padding:4px 14px;font-size:0.75em;color:#d2a8ff;margin:2px;}
  .stButton button{background:linear-gradient(135deg,#238636,#2ea043);color:white;
    border:none;border-radius:8px;padding:8px 20px;font-weight:600;transition:all 0.2s;}
  .stButton button:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(46,160,67,0.4);}
  .header-banner{background:linear-gradient(135deg,#0f3460,#16213e,#533483);
    padding:24px 32px;border-radius:16px;margin-bottom:20px;border:1px solid #30363d;}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1 style="color:#58a6ff;margin:0;font-size:2em;">📡 Spectrum-SLM — Phase 2</h1>
  <p style="color:#c9d1d9;margin:6px 0 10px;">New SDR Dataset · 5-Class Modulation Recognition</p>
  <span class="phase2-badge">BPSK</span><span class="phase2-badge">QPSK</span>
  <span class="phase2-badge">8PSK</span><span class="phase2-badge">16QAM</span>
  <span class="phase2-badge" style="background:linear-gradient(135deg,#533483,#21262d);">DQPSK ★ New</span>
  &nbsp;&nbsp;
  <span class="phase2-badge" style="background:#161b22;color:#8b949e;">IIT Palakkad</span>
  <span class="phase2-badge" style="background:#161b22;color:#8b949e;">~1M Parameters</span>
</div>
""", unsafe_allow_html=True)


# ─── Resolve best checkpoint path (Phase 3 > Phase 2) ────────────────────────
def _resolve_best_ckpt() -> str:
    """Phase 3 checkpoint has ALL heads trained (incl. generative).
    Fall back to Phase 2 (supervised only) if Phase 3 is missing."""
    p3 = os.path.join(CKPT_PHASE3, "slm_phase3_best.pt")
    p2 = os.path.join(CKPT_PHASE2, CKPT_PHASE2_BEST)
    if os.path.exists(p3):
        return p3
    return p2


# ─── Model + Normalizer loader ────────────────────────────────────────────────
@st.cache_resource
def load_model_and_normalizer(ckpt_path: str, norm_path: str):
    model = SpectrumSLM(
        n_bins=N_BINS, patch_size=1, d_model=128,
        nhead=4, num_layers=4, dim_feedforward=512,
        dropout=0.1, n_mod_classes=N_MOD_CLASSES_V2,
    )
    loaded_phase = None
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ck.get("model", ck)
        # Filter out keys whose tensor sizes don't match current model
        model_state = model.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = [k for k in state if k not in compatible]
        if skipped:
            print(f"[WARN] Skipped {len(skipped)} mismatched keys (old architecture):")
            for k in skipped[:6]:
                print(f"  {k}: ckpt={state[k].shape} vs model={model_state.get(k,'missing')}")
        model.load_state_dict(compatible, strict=False)
        # Detect which phase the checkpoint came from
        if "phase3" in ckpt_path:
            loaded_phase = 3
        elif "phase2" in ckpt_path:
            loaded_phase = 2
        else:
            loaded_phase = 2

    model.eval()

    scaler = None
    if norm_path and os.path.exists(norm_path):
        with open(norm_path, "rb") as f:
            scaler = pickle.load(f)

    return model, scaler, loaded_phase


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    ckpt_path = st.text_input(
        "Checkpoint path",
        value=_resolve_best_ckpt(),
    )
    norm_path = st.text_input(
        "Normalizer path",
        value=os.path.join(CKPT_PHASE2, NORMALIZER_FILE),
    )
    model, scaler, loaded_phase = load_model_and_normalizer(ckpt_path, norm_path)

    if loaded_phase == 3:
        st.success("✅ Phase 3 model loaded (all heads trained)")
    elif loaded_phase == 2:
        st.success("✅ Phase 2 model loaded (5-class, supervised)")
    else:
        st.warning("⚠️ Demo mode — untrained weights")
    if scaler:
        st.success("✅ Normalizer loaded")
    else:
        st.info("ℹ️ No normalizer — using raw PSD values")

    st.markdown("---")
    st.markdown("### 🔧 Model Info")
    n_params = sum(p.numel() for p in model.parameters())
    st.metric("Parameters",   f"{n_params/1e6:.2f}M")
    st.metric("Mod classes",  f"{N_MOD_CLASSES_V2} (with DQPSK)")
    st.metric("Patch tokens", "192 + CLS = 193")
    st.metric("Heads/Layers", "4 / 4")
    st.metric("N_BINS",       "192 (confirmed real)")
    st.markdown("---")
    st.caption("Anjani · Ashish Joshi · Mayank\nGuide: Dr. Abhinandan S.P.\nApril 2026")


# ─── Inference helper ─────────────────────────────────────────────────────────
def run_inference(psd_vec: np.ndarray) -> dict:
    p = psd_vec.reshape(1, -1).astype(np.float32)
    if scaler is not None:
        try:
            # Guard: only use scaler if it was fitted on the same number of features
            expected = getattr(scaler, 'n_features_in_', None) or getattr(
                getattr(scaler, 'scaler', None), 'n_features_in_', None)
            if expected is None or expected == p.shape[1]:
                p = scaler.transform(p).astype(np.float32)
            else:
                # Old normalizer (176-bin) vs current 192-bin — do simple z-score inline
                p = ((p - p.mean()) / (p.std() + 1e-8)).astype(np.float32)
        except Exception:
            p = ((p - p.mean()) / (p.std() + 1e-8)).astype(np.float32)

    t = torch.tensor(p, dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(t)
    lat = (time.perf_counter() - t0) * 1000
    pu_p  = torch.softmax(out["pu_logits"],  dim=1)[0].numpy()
    mod_p = torch.softmax(out["mod_logits"], dim=1)[0].numpy()
    snr   = float(np.clip(out["snr_pred"][0].item(), 0, 30))

    # Use the ACTUAL model output from the checkpoint
    gen_pred_np = out["gen_pred"][0].numpy()   # model's reconstruction, normalized space
    p_norm_np   = p[0]                         # what the model actually saw (normalized input)

    return {
        "pu_prob":     float(pu_p[1]),
        "pu_present":  bool(pu_p[1] > 0.5),
        "mod_probs":   mod_p.tolist(),
        "mod_pred":    int(np.argmax(mod_p)),
        "snr_db":      snr,
        "gen_psd":     gen_pred_np.tolist(),   # normalized — for chart
        "input_norm":  p_norm_np.tolist(),     # normalized input — for chart
        "latency_ms":  lat,
    }



def psd_fig(psd: np.ndarray, gen: np.ndarray = None, title: str = "PSD") -> go.Figure:
    freq = np.linspace(2380, 2420, N_BINS)   # 192 bins
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=psd, mode="lines", name="Input PSD",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"))
    if gen is not None:
        fig.add_trace(go.Scatter(x=freq, y=gen, mode="lines",
            name="Predicted Next PSD", line=dict(color="#3fb950", width=2, dash="dash")))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#c9d1d9")),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=280,
        xaxis=dict(title="Frequency (MHz)", color="#8b949e", gridcolor="#21262d"),
        yaxis=dict(title="Power (norm.)", color="#8b949e", gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22", font=dict(color="#c9d1d9")),
        font=dict(family="Inter"), margin=dict(l=40,r=20,t=40,b=40),
    )
    return fig


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_scan, tab_batch, tab_explore, tab_research = st.tabs([
    "🔭 Single Scan", "📊 Batch Analysis", "🗃️ Dataset Explorer", "📈 Research"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — Inference & Chat Assistant
# ════════════════════════════════════════════════════════════
with tab_scan:
    st.markdown("### 🔭 Inference & Chat Assistant")
    
    col_main, col_chat = st.columns([1.6, 1.2], gap="large")
    
    with col_main:
        st.markdown("#### 1. Spectrum Input & Prediction")
        col_ctrl, col_res = st.columns([1, 1.4])
        
        with col_ctrl:
            input_mode = st.radio("Input Mode", ["📤 Upload Real .pth File", "🎛️ Synthetic Generator"], horizontal=True)
            
            if input_mode == "📤 Upload Real .pth File":
                st.markdown("Upload a `.pth` file from `Symbol1/`, `Symbol2/`, or `Symbol3/` (e.g., inside the `bpsk/` or `qpsk/` folders).")
                uploaded_file = st.file_uploader("Choose a .pth file", type=["pth"])
                
                if uploaded_file is not None:
                    if st.button("🔍 Run Real Inference", use_container_width=True):
                        data = torch.load(uploaded_file, map_location="cpu", weights_only=False)

                        # ── Collect up to 100 samples across ALL SNR bins ──────────────
                        samples = []   # list of (psd_array, true_pu, true_snr)

                        def collect_from_pairs(pairs, bins, max_per_bin=8):
                            for b in sorted(bins, reverse=True):  # highest SNR first
                                if b not in pairs: continue
                                for entry in pairs[b][:max_per_bin]:
                                    psd_raw = np.array(entry[0], dtype=np.float32).flatten()
                                    pu_val  = int(np.array(entry[1]).flatten()[0]) if len(entry) > 1 else -1
                                    psd_arr = psd_raw[:N_BINS] if len(psd_raw) >= N_BINS else \
                                              np.pad(psd_raw, (0, N_BINS - len(psd_raw)))
                                    samples.append((psd_arr, pu_val, float(b)))
                                    if len(samples) >= 100:
                                        return

                        if isinstance(data, dict) and 'pairs_by_bin' in data and 'bins' in data:
                            collect_from_pairs(data['pairs_by_bin'], data['bins'])

                        # Fallback: recursive search for one PSD
                        if not samples:
                            def find_psd(obj):
                                if isinstance(obj, (np.ndarray,)) or torch.is_tensor(obj):
                                    flat = np.array(obj.cpu() if torch.is_tensor(obj) else obj).flatten()
                                    if len(flat) >= 100: return flat
                                elif isinstance(obj, (list, tuple)):
                                    for item in obj:
                                        r = find_psd(item)
                                        if r is not None: return r
                                elif isinstance(obj, dict):
                                    for k in ['psd','power','features','spectrum']:
                                        if k in obj:
                                            r = find_psd(obj[k])
                                            if r is not None: return r
                                    for v in obj.values():
                                        r = find_psd(v)
                                        if r is not None: return r
                                return None
                            raw = find_psd(data)
                            if raw is not None:
                                arr = raw[:N_BINS] if len(raw) >= N_BINS else np.pad(raw, (0, N_BINS - len(raw)))
                                samples.append((arr.astype(np.float32), -1, None))

                        if not samples:
                            st.error("Could not find any PSD data in this .pth file.")
                        else:
                            # ── Batch inference ────────────────────────────────────────
                            pu_preds, pu_probs, mod_preds, snr_preds = [], [], [], []
                            true_pus, true_snrs = [], []

                            for psd_arr, tpu, tsnr in samples:
                                r = run_inference(psd_arr)
                                pu_preds.append(int(r["pu_present"]))
                                pu_probs.append(r["pu_prob"])
                                mod_preds.append(r["mod_pred"])
                                snr_preds.append(r["snr_db"])
                                if tpu >= 0:  true_pus.append(tpu)
                                if tsnr is not None: true_snrs.append(tsnr)

                            # ── Aggregate results ──────────────────────────────────────
                            mean_pu_prob  = float(np.mean(pu_probs))
                            pu_detected   = mean_pu_prob > 0.5
                            mean_snr      = float(np.mean(snr_preds))
                            from collections import Counter
                            top_mod       = Counter(mod_preds).most_common(1)[0][0]
                            top_mod_pct   = 100 * Counter(mod_preds).most_common(1)[0][1] / len(mod_preds)

                            # Accuracy vs ground truth
                            acc_str = ""
                            if true_pus:
                                correct = sum(p == t for p, t in zip(pu_preds, true_pus))
                                acc_str = f"\n- **PU Accuracy on this file: {correct}/{len(true_pus)} = {100*correct/len(true_pus):.1f}%**"
                            gt_snr_str = f"{np.mean(true_snrs):.1f} dB" if true_snrs else "unknown"

                            # Use highest-SNR sample's PSD for display
                            best_psd = samples[0][0]
                            res_display = run_inference(best_psd)

                            st.session_state.p2_psd = best_psd
                            st.session_state.p2_res = res_display

                            # Override display metrics with batch aggregate
                            st.session_state.p2_res["pu_prob"]    = mean_pu_prob
                            st.session_state.p2_res["pu_present"] = pu_detected
                            st.session_state.p2_res["snr_db"]     = mean_snr
                            st.session_state.p2_res["mod_pred"]   = top_mod

                            pu_text = (f"Primary User **DETECTED** ({mean_pu_prob*100:.1f}% avg confidence)"
                                       if pu_detected else
                                       f"NO Primary User ({(1-mean_pu_prob)*100:.1f}% avg confidence of absence)")

                            chat_msg = (f"⚡ **Batch Inference on {len(samples)} samples from file:**\n"
                                        f"- {pu_text}\n"
                                        f"- Modulation: **{MOD_NAMES_V2[top_mod]}** ({top_mod_pct:.0f}% of samples)\n"
                                        f"- Mean Estimated SNR: **{mean_snr:.1f} dB**"
                                        f"{acc_str}\n"
                                        f"- Ground Truth SNR: {gt_snr_str}\n\n"
                                        f"*What would you like to know about this file?*")

                            if "messages" not in st.session_state:
                                st.session_state.messages = []
                            st.session_state.messages.append({"role": "assistant", "content": chat_msg})

            
            else:
                snr_t  = st.slider("Target SNR (dB)", 3.0, 20.0, 10.0, 0.5)
                pu_sel = st.selectbox("PU Status", ["Present (PU=1)", "Absent (PU=0)"])
                mod_sel= st.selectbox("Modulation", MOD_NAMES_V2)
                is_pu  = pu_sel == "Present (PU=1)"
                mod_id = MOD_NAMES_V2.index(mod_sel)
    
                if st.button("🔍 Run Synthetic Inference", use_container_width=True):
                    freqs = np.linspace(-1, 1, N_BINS)
                    widths = [0.20, 0.25, 0.30, 0.35, 0.22]   # per-mod bandwidth
                    psd = np.random.randn(N_BINS).astype(np.float32) * 1.5 - 22.0
                    if is_pu:
                        psd += (snr_t * 0.8) * np.exp(
                            -freqs**2 / (2 * widths[mod_id]**2)).astype(np.float32)
                    st.session_state.p2_psd = psd
                    st.session_state.p2_res = run_inference(psd)
                    st.session_state.p2_true_pu  = is_pu
                    st.session_state.p2_true_mod = mod_id
                    
                    # Auto-generate an initial thought from the assistant upon new scan
                    res = st.session_state.p2_res
                    pu_text = f"Primary User was **DETECTED** ({res['pu_prob']*100:.1f}% confidence)" if res["pu_present"] else f"NO Primary User found ({(1-res['pu_prob'])*100:.1f}% confidence of absence)"
                    mod_text = MOD_NAMES_V2[res['mod_pred']]
                    mod_conf = res['mod_probs'][res['mod_pred']] * 100
                    
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                        
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚡ **Synthetic Scan Analyzed:**\n- {pu_text}\n- Modulation: {mod_text} ({mod_conf:.1f}%)\n- Estimated SNR: {res['snr_db']:.1f} dB\n\n*Keep in mind, synthetic signals lack real hardware phase shifts. What would you like to know?*"
                    })

        with col_res:
            if "p2_res" in st.session_state:
                res = st.session_state.p2_res
                # Use the normalized input and normalized prediction for a perfect 1:1 scale match
                psd_norm = np.array(res.get("input_norm", st.session_state.p2_psd))
                gen_norm = np.array(res.get("gen_psd", []))
                
                # Small visualization
                fig_small = psd_fig(psd_norm, gen_norm if len(gen_norm) else None, "Processed Snapshot (Normalized)")
                fig_small.update_layout(height=220, margin=dict(l=20,r=10,t=30,b=20))
                st.plotly_chart(fig_small, use_container_width=True)

                m1,m2 = st.columns(2)
                pu_col = "#3fb950" if res["pu_present"] else "#f85149"
                m1.markdown(f"""<div class="metric-card" style="padding:10px;">
                  <div class="metric-value" style="color:{pu_col};font-size:1.5em;">
                    {'✅ YES' if res['pu_present'] else '⛔ NO'}
                  </div>
                  <div class="metric-label">PU ({res['pu_prob']*100:.1f}%)</div>
                </div>""", unsafe_allow_html=True)
                
                m2.markdown(f"""<div class="metric-card" style="padding:10px;">
                  <div class="metric-value" style="color:#ffa657;font-size:1.5em;">
                    {MOD_NAMES_V2[res['mod_pred']]}
                  </div>
                  <div class="metric-label">SNR: {res['snr_db']:.1f} dB</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Configure input and click **Run Inference** to view results.")

    with col_chat:
        st.markdown("#### 💬 Interactive AI Assistant")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I am your Spectrum-SLM Assistant. Run an inference scan on the left, and I'll help you analyze the results contextually. What would you like to know?"}
            ]
            
        # Display chat messages (Streamlit 1.24+ style chat container)
        chat_container = st.container(height=380)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
        # Intelligent Response Generator
        def generate_ai_response(prompt):
            prompt = prompt.lower()
            ctx = st.session_state.get("p2_res")
            
            # Casual conversation
            if prompt in ["hi", "hello", "hey"]:
                return "Hello! How can I assist you with spectrum sensing today?"
            if "who are you" in prompt or "what are you" in prompt:
                return "I am the Spectrum-SLM AI Assistant. I use a Small Language Model architecture to analyze radio frequencies, detect primary users, classify modulations, and estimate SNR in real time."
                
            # Need context for the rest
            if not ctx:
                return "Please run a spectrum inference scan on the left before asking for an analysis. I need live data to provide meaningful insights!"
                
            pu = ctx["pu_present"]
            pu_prob = ctx["pu_prob"] * 100
            mod_pred = MOD_NAMES_V2[ctx["mod_pred"]]
            mod_prob = ctx["mod_probs"][ctx["mod_pred"]] * 100
            snr = ctx["snr_db"]
            lat = ctx["latency_ms"]
            
            if any(w in prompt for w in ["explain", "result", "analysis", "what happened", "summary"]):
                pu_text = f"detected a Primary User with {pu_prob:.1f}% confidence" if pu else f"found NO Primary User ({100-pu_prob:.1f}% confidence)"
                ans = f"Based on the latest scan, my model **{pu_text}**.\n\n"
                if pu:
                    ans += f"The signal's transmission scheme is classified as **{mod_pred}** (confidence: {mod_prob:.1f}%) with an estimated channel SNR of **{snr:.1f} dB**."
                else:
                    ans += f"Although there is no active transmission, the background noise structure marginally resembles a **{mod_pred}** signature at a very low **{snr:.1f} dB**."
                ans += f"\n\n*This inference was processed in just {lat:.1f} ms.*"
                return ans
                
            if any(w in prompt for w in ["reliable", "confidence", "trust", "sure", "accurate"]):
                if pu_prob > 90 or pu_prob < 10:
                    return f"Yes, the model is **highly confident** (PU probability is {pu_prob:.1f}%). The estimated SNR of {snr:.1f} dB provides a strong enough signal-to-noise ratio for reliable boundary decision-making."
                else:
                    return f"The model is mathematically **uncertain** (PU probability is {pu_prob:.1f}%). This ambiguity is common when the SNR drops to {snr:.1f} dB, pushing the signal close to the noise floor. I recommend observing another packet."
                    
            if any(w in prompt for w in ["modulation", "type", "scheme", "encode"]):
                return f"The detected modulation scheme is **{mod_pred}** with a calculated probability of {mod_prob:.1f}%. My transformer encoder categorizes the spectral patches into one of five learned bases: BPSK, QPSK, 8PSK, 16QAM, or DQPSK."
                
            if any(w in prompt for w in ["snr", "signal", "noise"]):
                return f"The estimated Signal-to-Noise Ratio (SNR) is **{snr:.1f} dB**. This is calculated simultaneously via the Multi-Task learning phase, actively measuring the channel's noise floor against the signal's peak."
                
            if any(w in prompt for w in ["next", "future", "predict", "forecast"]):
                return "Using our Phase 3 Generative Head, I autoregressively forecasted the **next upcoming spectrum state** (represented by the dashed green line in the plot). This provides foresight into spectrum occupancy before it physically occurs."

            return f"I see you're referring to the latest {mod_pred} signal scan at {snr:.1f} dB. Could you be more specific? I can explain the general results, analyze confidence metrics, or explain the generative forecasting."

        # Chat Input
        if prompt := st.chat_input("Ask me to analyze the results..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
            # Generate and add assistant response
            response = generate_ai_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)


# ════════════════════════════════════════════════════════════
# TAB 2 — Batch Analysis
# ════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📊 Batch Spectrum Analysis (5-class)")
    from sklearn.metrics import accuracy_score, mean_absolute_error
    n_samples = st.slider("Synthetic samples", 100, 2000, 500, 100)
    if st.button("▶ Run Batch Analysis", use_container_width=False):
        with st.spinner("Running batch inference …"):
            rng   = np.random.default_rng(42)
            psds_ = []; pu_ = []; mod_ = []; snr_ = []
            freqs = np.linspace(-1, 1, N_BINS)
            widths = [0.20, 0.25, 0.30, 0.35, 0.22]
            for _ in range(n_samples):
                pu  = rng.integers(0, 2)
                mod = rng.integers(0, N_MOD_CLASSES_V2)
                snr = rng.uniform(3, 20) if pu else rng.uniform(3, 8)
                p   = (rng.standard_normal(N_BINS) * 1.5 - 22.0).astype(np.float32)
                if pu:
                    p += (snr * 0.8) * np.exp(-freqs**2/(2*widths[mod]**2)).astype(np.float32)
                psds_.append(p); pu_.append(int(pu)); mod_.append(int(mod)); snr_.append(float(snr))

            pu_pred = []; mod_pred = []; snr_pred = []
            t0 = time.perf_counter()
            for p in psds_:
                r = run_inference(p)
                pu_pred.append(int(r["pu_present"]))
                mod_pred.append(r["mod_pred"])
                snr_pred.append(r["snr_db"])
            total_ms = (time.perf_counter() - t0) * 1000

        pu_a  = accuracy_score(pu_, pu_pred)
        mod_a = accuracy_score(mod_, mod_pred)
        snr_m = mean_absolute_error(snr_, snr_pred)

        r1,r2,r3,r4 = st.columns(4)
        r1.metric("PU Accuracy",  f"{pu_a*100:.2f}%")
        r2.metric("Mod Accuracy", f"{mod_a*100:.2f}%")
        r3.metric("SNR MAE",      f"{snr_m:.2f} dB")
        r4.metric("Throughput",   f"{n_samples/(total_ms/1000):.0f} samp/s")

        # Per-SNR PU accuracy
        st.markdown("#### PU Detection Accuracy vs SNR")
        snr_arr = np.array(snr_); pu_arr = np.array(pu_); pp_arr = np.array(pu_pred)
        bins = list(range(3,22,2)); accs = []
        for b in bins:
            mask = (snr_arr>=b-1)&(snr_arr<b+1)
            accs.append(accuracy_score(pu_arr[mask],pp_arr[mask])*100 if mask.sum()>3 else None)
        valid = [(b,a) for b,a in zip(bins,accs) if a is not None]
        if valid:
            vb,va = zip(*valid)
            snr_fig = go.Figure()
            snr_fig.add_trace(go.Scatter(x=list(vb),y=list(va),
                mode="lines+markers",name="PU Acc",
                line=dict(color="#58a6ff",width=2),marker=dict(size=8)))
            snr_fig.add_hline(y=90,line_dash="dash",line_color="#3fb950",
                              annotation_text="90% target")
            snr_fig.update_layout(
                xaxis_title="SNR (dB)", yaxis_title="PU Accuracy (%)",
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                yaxis=dict(range=[40,105],color="#8b949e",gridcolor="#21262d"),
                xaxis=dict(color="#8b949e"),
                height=280, font=dict(color="#c9d1d9",family="Inter"))
            st.plotly_chart(snr_fig, use_container_width=True)

        # Per-class modulation accuracy
        st.markdown("#### Per-Modulation Accuracy")
        mod_arr = np.array(mod_); mp_arr = np.array(mod_pred)
        mod_accs = []
        for i,n in enumerate(MOD_NAMES_V2):
            mask = (mod_arr == i)
            acc  = accuracy_score(mod_arr[mask], mp_arr[mask])*100 if mask.sum()>0 else 0
            mod_accs.append(acc)
        mbar = go.Figure(go.Bar(x=MOD_NAMES_V2, y=mod_accs, marker_color=MOD_COLORS_V2,
            text=[f"{a:.1f}%" for a in mod_accs], textposition="outside"))
        mbar.update_layout(plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",height=220,
            yaxis=dict(range=[0,115],color="#8b949e"),xaxis=dict(color="#8b949e"),
            font=dict(color="#c9d1d9",family="Inter"),margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(mbar, use_container_width=True)
    else:
        st.info("Click **Run Batch Analysis** to begin.")


# ════════════════════════════════════════════════════════════
# TAB 3 — Dataset Explorer
# ════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("### 🗃️ Real Dataset Explorer")
    st.caption(f"Source 1: `{SECONDARY_USER_DIR}`  |  Source 2: `{NEW_DATASET_DIR}`")

    col_e1, col_e2 = st.columns([1, 3])
    with col_e1:
        sel_mod = st.selectbox("Modulation", MOD_NAMES_V2, key="exp_mod")
        sel_sym = st.selectbox("Symbol dir", ["Symbol2","Symbol3"], key="exp_sym")
        n_show  = st.slider("Samples to show", 5, 100, 20)
        if st.button("📂 Load Real PSD Samples"):
            mod_folder = {"BPSK":"bpsk","QPSK":"qpsk","8PSK":"8psk",
                          "16QAM":"16qam","DQPSK":"dqpsk"}[sel_mod]
            dset_path = os.path.join(NEW_DATASET_DIR, sel_sym, mod_folder, "dataset.pth")
            if os.path.exists(dset_path):
                raw = torch.load(dset_path, map_location="cpu", weights_only=False)
                psds_r = np.array(raw["psds"]).squeeze(-1) if np.array(raw["psds"]).ndim==3 else np.array(raw["psds"])
                snrs_r = np.array(raw["snrs"])
                pu_r   = np.array(raw.get("pu_flags", raw.get("pu_labels", [])))
                df_exp = pd.DataFrame({"snr_db": snrs_r, "pu": pu_r,
                                       "psd_mean": psds_r.mean(axis=1),
                                       "psd_std":  psds_r.std(axis=1)})
                st.session_state.exp_df   = df_exp
                st.session_state.exp_psds = psds_r
                st.session_state.exp_mod  = sel_mod
                st.success(f"Loaded {len(df_exp):,} real samples — PSD shape: {psds_r.shape}")
            else:
                st.warning(f"File not found: {dset_path}")

    with col_e2:
        if "exp_df" in st.session_state:
            df       = st.session_state.exp_df
            psds_exp = st.session_state.exp_psds
            mn       = st.session_state.exp_mod
            col_d1, col_d2 = st.columns(2)
            col_d1.metric("Samples", f"{len(df):,}")
            col_d2.metric("PU=1", f"{df['pu'].sum():,} ({df['pu'].mean()*100:.1f}%)")
            st.dataframe(df.head(n_show), use_container_width=True)

            # SNR distribution
            st.markdown(f"#### SNR Distribution — {mn}")
            fig_snr = go.Figure(go.Histogram(
                x=df["snr_db"], nbinsx=40,
                marker_color=MOD_COLORS_V2[MOD_NAMES_V2.index(mn)]))
            fig_snr.update_layout(plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",
                height=220, xaxis=dict(title="SNR (dB)",color="#8b949e"),
                yaxis=dict(color="#8b949e"),font=dict(color="#c9d1d9",family="Inter"),
                margin=dict(l=30,r=10,t=10,b=30))
            st.plotly_chart(fig_snr, use_container_width=True)

            # Sample PSD
            st.markdown("#### Sample PSD Vector (192 bins)")
            sample_idx = st.slider("Sample index", 0, min(len(psds_exp)-1,999), 0)
            freq = np.linspace(2380, 2420, N_BINS)
            fig_p = go.Figure(go.Scatter(x=freq, y=psds_exp[sample_idx],
                mode="lines", line=dict(color=MOD_COLORS_V2[MOD_NAMES_V2.index(mn)],width=1.5),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.05)"))
            fig_p.update_layout(plot_bgcolor="#0d1117",paper_bgcolor="#0d1117",height=220,
                xaxis=dict(title="Freq (MHz)",color="#8b949e"),
                yaxis=dict(title="Power",color="#8b949e"),
                font=dict(color="#c9d1d9",family="Inter"),margin=dict(l=30,r=10,t=10,b=30))
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Select modulation + symbol dir and click **Load Real PSD Samples**.")


# ════════════════════════════════════════════════════════════
# TAB 4 — Research / Metrics
# ════════════════════════════════════════════════════════════
with tab_research:
    st.markdown("### 📈 Phase 2 Training Results")
    metrics_path = os.path.join(CKPT_PHASE2, METRICS_FILE)
    history_path = os.path.join(CKPT_PHASE2, "training_history_phase2.json")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            st.markdown("#### 📊 Test Metrics")
            mc1,mc2,mc3 = st.columns(3)
            mc1.metric("PU Accuracy",  f"{m.get('pu_accuracy',0)*100:.2f}%")
            mc2.metric("Mod Accuracy", f"{m.get('mod_accuracy',0)*100:.2f}%")
            mc3.metric("SNR MAE",      f"{m.get('snr_mae_db',0):.3f} dB")
            mc1.metric("PU F1",   f"{m.get('pu_f1',0):.4f}")
            mc2.metric("Mod F1",  f"{m.get('mod_f1_macro',0):.4f}")
            mc3.metric("PU AUC",  f"{m.get('pu_auc',0):.4f}")
            with st.expander("Full metrics JSON"):
                st.json(m)
        else:
            st.info("No metrics file yet — train the model first.\n\n"
                    f"Expected: `{metrics_path}`")

    with col_m2:
        if os.path.exists(history_path):
            with open(history_path) as f:
                hist = json.load(f)
            epochs_h = [h["epoch"] for h in hist]
            tr_h = [h.get("train_total", h.get("train",0)) for h in hist]
            vl_h = [h.get("val_total",  h.get("val",  0)) for h in hist]
            hfig = go.Figure()
            hfig.add_trace(go.Scatter(x=epochs_h, y=tr_h, name="Train Loss",
                line=dict(color="#58a6ff",width=2)))
            hfig.add_trace(go.Scatter(x=epochs_h, y=vl_h, name="Val Loss",
                line=dict(color="#3fb950",width=2,dash="dash")))
            hfig.update_layout(
                title="Training History", plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117", height=300,
                xaxis=dict(title="Epoch",color="#8b949e",gridcolor="#21262d"),
                yaxis=dict(title="Loss",color="#8b949e",gridcolor="#21262d"),
                legend=dict(bgcolor="#161b22",font=dict(color="#c9d1d9")),
                font=dict(color="#c9d1d9",family="Inter"))
            st.plotly_chart(hfig, use_container_width=True)
        else:
            st.info("No training history yet.")

    # Predictions CSV viewer
    pred_path = os.path.join(CKPT_PHASE2, "predictions_phase2.csv")
    if os.path.exists(pred_path):
        st.markdown("#### 🗒️ Sample Predictions (test set)")
        df_pred = pd.read_csv(pred_path).head(50)
        st.dataframe(df_pred, use_container_width=True)

    st.markdown("#### 🆚 Phase 2 vs Phase 1 (Original Dataset)")
    comp = {
        "Metric": ["Mod Classes", "Dataset", "PU Acc (target)", "Mod Acc (target)", "SNR MAE (target)"],
        "Phase 1 (Original)": ["4 (no DQPSK)", "Secondary_User/", "97–98%", "92–95%", "<1.5 dB"],
        "Phase 2 (New)":      ["5 (+ DQPSK)", "Symbol1/2/3", "~97%", "~92%", "<1.5 dB"],
    }
    st.dataframe(pd.DataFrame(comp), use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:0.8em;'>"
    "Spectrum-SLM Phase 2 &nbsp;|&nbsp; Anjani · Ashish Joshi · Mayank "
    "&nbsp;|&nbsp; Guide: Dr. Abhinandan S.P. &nbsp;|&nbsp; April 2026 &nbsp;|&nbsp; IIT Palakkad"
    "</p>",
    unsafe_allow_html=True,
)
