import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SST Harmonic Distortion Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #e6edf3; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    /* Section header */
    .section-header {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
        padding: 18px 28px;
        border-radius: 12px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(31,111,235,0.35);
    }
    .section-header h1, .section-header h2 {
        margin: 0; color: white; font-weight: 700;
    }

    /* Metric card */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #1f6feb;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .metric-card h3 { color: #58a6ff; margin: 0 0 4px 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card p  { color: #e6edf3; margin: 0; font-size: 1.4rem; font-weight: 700; }

    /* Info box */
    .info-box {
        background: #0d2137;
        border: 1px solid #1f6feb;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 10px 0;
    }
    .info-box h4 { color: #58a6ff; margin-top: 0; }
    .info-box p, .info-box li { color: #c9d1d9; line-height: 1.7; }

    /* Advantage card */
    .adv-card {
        background: #0f2b1a;
        border: 1px solid #2ea043;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .adv-card h4 { color: #3fb950; margin: 0 0 6px 0; }
    .adv-card p  { color: #c9d1d9; margin: 0; font-size: 0.95rem; }

    /* App card */
    .app-card {
        background: #1a1228;
        border: 1px solid #8957e5;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .app-card h4 { color: #d2a8ff; margin: 0 0 6px 0; }
    .app-card p  { color: #c9d1d9; margin: 0; font-size: 0.95rem; }

    /* Member card */
    .member-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 6px 0;
    }
    .member-avatar {
        width: 44px; height: 44px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1f6feb, #8957e5);
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; color: white; font-size: 1.1rem;
        flex-shrink: 0;
    }
    .member-info h4 { color: #e6edf3; margin: 0; font-size: 1rem; }
    .member-info p  { color: #8b949e; margin: 0; font-size: 0.85rem; }

    /* Nav pill */
    .nav-title {
        color: #58a6ff;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        padding: 8px 0 4px 0;
    }

    /* Title page */
    .title-hero {
        background: linear-gradient(135deg, #0d2137 0%, #1a1228 50%, #0f2b1a 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 50px 40px;
        text-align: center;
        margin-bottom: 30px;
    }
    .title-hero h1 {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        line-height: 1.3;
        margin-bottom: 12px;
    }
    .title-hero .subtitle {
        color: #58a6ff;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }
    .title-badge {
        display: inline-block;
        background: rgba(31,111,235,0.15);
        border: 1px solid #1f6feb;
        color: #58a6ff;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 6px 4px;
    }

    hr { border-color: #30363d; }
    h3 { color: #58a6ff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_sim_data():
    df = pd.read_csv("simdata.csv")
    df.columns = ["Time", "Signal_1", "Signal_2", "Signal_3"]
    return df

@st.cache_data
def load_students():
    df = pd.read_csv("STUDENTS.CSV")
    df.columns = [c.strip().lower() for c in df.columns]
    # normalise to 'name' and 'rollno' regardless of original casing
    df = df.rename(columns={"name": "Name", "rollno": "RollNo"})
    return df

df = load_sim_data()
students_df = load_students()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def compute_thd(signal, fs):
    N = len(signal)
    yf = np.abs(fft(signal)) / N
    xf = fftfreq(N, 1 / fs)
    pos = xf > 0
    xf, yf = xf[pos], yf[pos]
    # Find fundamental (highest amplitude peak)
    peaks, _ = find_peaks(yf, height=np.max(yf) * 0.05)
    if len(peaks) == 0:
        return 0.0, xf, yf
    fund_idx = peaks[np.argmax(yf[peaks])]
    fund_amp = yf[fund_idx]
    # Sum harmonics (2nd–10th)
    harm_sum_sq = 0
    for n in range(2, 11):
        target_f = xf[fund_idx] * n
        idx = np.argmin(np.abs(xf - target_f))
        harm_sum_sq += yf[idx] ** 2
    thd = (np.sqrt(harm_sum_sq) / fund_amp) * 100
    return thd, xf, yf

fs = 1 / (df["Time"].iloc[1] - df["Time"].iloc[0])  # ~20000 Hz


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ SST Analysis")
    st.markdown("---")
    st.markdown('<div class="nav-title">Navigation</div>', unsafe_allow_html=True)

    pages = {
        "🏠  Title & Topic":      "title",
        "👥  Group Members":      "members",
        "📖  Introduction":       "intro",
        "🔌  What is SST?":       "what_sst",
        "⚙️   Working Principle":  "working",
        "✅  Advantages":         "advantages",
        "🏭  Applications":       "applications",
    }
    selection = st.radio("", list(pages.keys()), label_visibility="collapsed")
    page = pages[selection]

    st.markdown("---")
    st.markdown(
        '<div style="color:#8b949e;font-size:0.78rem;line-height:1.6;">'
        'Department of Electrical Engineering<br>'
        'Harmonic Distortion in SST<br>'
        'MATLAB Simulink Simulation'
        '</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — TITLE
# ═══════════════════════════════════════════════════════════════
if page == "title":
    st.markdown("""
    <div class="title-hero">
        <div style="font-size:3rem;margin-bottom:12px;">⚡</div>
        <h1>Harmonic Distortion Analysis in<br>Solid-State Transformers</h1>
        <p class="subtitle">and Its Impact on Power Quality</p>
        <br>
        <span class="title-badge">Electrical Engineering</span>
        <span class="title-badge">Power Electronics</span>
        <span class="title-badge">MATLAB Simulink</span>
        <span class="title-badge">Power Quality</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Simulation Duration</h3>
            <p>0.2 seconds</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Data Points</h3>
            <p>4,001 samples</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Sampling Rate</h3>
            <p>20,000 Hz</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### About This Presentation")
    st.markdown("""
    <div class="info-box">
        <h4>Project Overview</h4>
        <p>
        This presentation explores <strong>harmonic distortion</strong> in
        <strong>Solid-State Transformers (SST)</strong> — next-generation power conversion
        devices that replace conventional iron-core transformers using high-frequency
        power electronics. Our MATLAB Simulink simulation models a three-phase SST system
        and captures the three-phase output signals to analyze harmonic content and its
        impact on power quality.
        </p>
        <ul>
            <li>Three-phase SST modeled in MATLAB/Simulink with PWM switching</li>
            <li>Signal data exported and analyzed for harmonic spectrum</li>
            <li>THD (Total Harmonic Distortion) calculated per IEEE 519 standards</li>
            <li>Impact on power quality assessed through FFT analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — GROUP MEMBERS
# ═══════════════════════════════════════════════════════════════
elif page == "members":
    st.markdown("""
    <div class="section-header">
        <h2>👥 Group Members</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Total Members: {len(students_df)}**")
    st.markdown("")

    col1, col2 = st.columns(2)
    for i, row in students_df.iterrows():
        name = str(row["Name"]).strip()
        roll = str(row["RollNo"]).strip()
        initials = "".join([p[0] for p in name.split()][:2]).upper()
        card = f"""
        <div class="member-card">
            <div class="member-avatar">{initials}</div>
            <div class="member-info">
                <h4>{name}</h4>
                <p>Roll No: {roll}</p>
            </div>
        </div>"""
        if i % 2 == 0:
            col1.markdown(card, unsafe_allow_html=True)
        else:
            col2.markdown(card, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — INTRODUCTION
# ═══════════════════════════════════════════════════════════════
elif page == "intro":
    st.markdown("""
    <div class="section-header">
        <h2>📖 Introduction</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4>Power Quality in Modern Electrical Systems</h4>
        <p>
        Power quality refers to the characteristics of the electrical supply that allows
        equipment to function properly. With the rise of power electronics in industrial,
        commercial, and renewable energy systems, <strong>harmonic distortion</strong> has
        become one of the most critical power quality issues in modern grids.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>What is Harmonic Distortion?</h4>
            <p>
            Harmonic distortion occurs when non-linear loads draw current in a non-sinusoidal
            manner, injecting currents at frequencies that are integer multiples of the
            fundamental frequency (50 Hz or 60 Hz). These harmonics cause:
            </p>
            <ul>
                <li>Overheating of cables and transformers</li>
                <li>Malfunction of sensitive equipment</li>
                <li>Increased losses in distribution systems</li>
                <li>Interference with communication systems</li>
                <li>Reduced power factor</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>Why Study SST Harmonics?</h4>
            <p>
            Solid-State Transformers operate using high-frequency PWM switching converters,
            which are inherently non-linear. This switching action introduces harmonic
            components across a wide frequency range. Key motivations for this study:
            </p>
            <ul>
                <li>SSTs are core components in smart grids and EV charging</li>
                <li>Their switching harmonics can degrade grid power quality</li>
                <li>IEEE 519 standard mandates THD &lt; 5% for grid-tied devices</li>
                <li>Understanding harmonic content helps design better filters</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Objective of This Study")
    st.markdown("""
    <div class="info-box">
        <p>
        Using a <strong>MATLAB/Simulink</strong> simulation of a three-phase Solid-State
        Transformer, this study:
        </p>
        <ol>
            <li><strong>Models</strong> the SST circuit including the AC-DC rectifier stage,
                high-frequency DC-DC converter with isolation transformer, and DC-AC inverter output stage.</li>
            <li><strong>Simulates</strong> the three-phase output waveforms over 0.2 seconds at 20 kHz sampling.</li>
            <li><strong>Analyzes</strong> harmonic content via FFT (Fast Fourier Transform).</li>
            <li><strong>Calculates</strong> Total Harmonic Distortion (THD) for each phase.</li>
            <li><strong>Evaluates</strong> the impact on power quality against IEEE 519 limits.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — WHAT IS SST?
# ═══════════════════════════════════════════════════════════════
elif page == "what_sst":
    st.markdown("""
    <div class="section-header">
        <h2>🔌 What is a Solid-State Transformer (SST)?</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4>Definition</h4>
        <p>
        A <strong>Solid-State Transformer (SST)</strong>, also known as a Power Electronic
        Transformer (PET) or Smart Transformer, is an advanced power conversion device that
        replaces the conventional laminated-core transformer using <strong>power semiconductor
        switches</strong>, <strong>high-frequency magnetic cores</strong>, and
        <strong>sophisticated control algorithms</strong>. It performs the same voltage
        transformation and galvanic isolation function but with greatly enhanced capabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Key Differences from Conventional Transformer")
        features = [
            ("Operating Frequency", "50/60 Hz", "1 kHz – 100 kHz"),
            ("Core Material", "Silicon Steel Laminations", "Ferrite / Nanocrystalline"),
            ("Size & Weight", "Very Large & Heavy", "Compact & Lightweight"),
            ("Voltage Regulation", "Fixed Ratio", "Fully Controllable"),
            ("Power Flow", "Unidirectional", "Bidirectional"),
            ("Reactive Power", "Cannot Compensate", "Active Compensation"),
            ("Fault Isolation", "Slow (breaker-based)", "Fast (semiconductor)"),
        ]
        comp_df = pd.DataFrame(features, columns=["Feature", "Conventional Transformer", "Solid-State Transformer"])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Three-Stage Architecture")
        st.markdown("""
        <div class="info-box">
            <h4>Stage 1 — AC-DC Rectifier (Input Stage)</h4>
            <p>Converts the medium-voltage AC input to a DC bus using a PWM-controlled
            Active Front End (AFE) rectifier. Provides power factor correction and
            injects minimal harmonics back to the grid.</p>
        </div>
        <div class="info-box">
            <h4>Stage 2 — Isolated DC-DC Converter (Middle Stage)</h4>
            <p>A Dual Active Bridge (DAB) or Series Resonant Converter operates at
            high frequency (kHz range) with a compact high-frequency transformer
            providing galvanic isolation and voltage transformation.</p>
        </div>
        <div class="info-box">
            <h4>Stage 3 — DC-AC Inverter (Output Stage)</h4>
            <p>Converts the low-voltage DC to high-quality AC for distribution loads.
            PWM switching in this stage is the primary source of harmonic distortion
            in the output waveform.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Simulation Circuit Block Diagram")
    st.markdown("""
    <div class="info-box">
        <p>
        Our MATLAB/Simulink model implements the following chain:<br><br>
        <strong>Three-Phase Source → PWM Generator (2-Level) → AC-DC Converter →
        DC-DC Isolated Stage (Linear Transformer) → DC-AC Inverter →
        Three-Phase Series RLC Load → simdata output</strong>
        </p>
        <p>The simulation uses a discrete solver at <strong>5×10⁻⁵ s (50 µs) timestep</strong>,
        capturing three-phase output signals (Signal 1, 2, 3) which represent the
        phase voltages/currents at the output of the SST.</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — WORKING PRINCIPLE & GRAPHS
# ═══════════════════════════════════════════════════════════════
elif page == "working":
    st.markdown("""
    <div class="section-header">
        <h2>⚙️ Working Principle & Simulation Results</h2>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ──────────────────────────────
    with st.expander("Plot Controls", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            t_start = st.slider("Time Start (s)", 0.0, 0.18, 0.0, 0.005)
        with c2:
            t_end = st.slider("Time End (s)", 0.02, 0.2, 0.2, 0.005)
        with c3:
            signals = st.multiselect(
                "Signals to Display",
                ["Signal 1 (Phase A)", "Signal 2 (Phase B)", "Signal 3 (Phase C)"],
                default=["Signal 1 (Phase A)", "Signal 2 (Phase B)", "Signal 3 (Phase C)"],
            )

    mask = (df["Time"] >= t_start) & (df["Time"] <= t_end)
    dff = df[mask]

    sig_map = {
        "Signal 1 (Phase A)": ("Signal_1", "#2196F3"),
        "Signal 2 (Phase B)": ("Signal_2", "#FF5722"),
        "Signal 3 (Phase C)": ("Signal_3", "#FFC107"),
    }

    # ── Time-domain plot ─────────────────────
    st.markdown("#### Three-Phase Output Waveforms (Time Domain)")
    fig_time = go.Figure()
    for label in signals:
        col_name, color = sig_map[label]
        fig_time.add_trace(go.Scatter(
            x=dff["Time"], y=dff[col_name],
            name=label, line=dict(color=color, width=1.2),
            mode="lines",
        ))
    fig_time.add_hline(y=0, line_color="#555", line_width=1, line_dash="dot")
    fig_time.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        legend=dict(orientation="h", y=1.08),
        height=420,
        margin=dict(t=40, b=40),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # ── FFT / Harmonic Spectrum ──────────────
    st.markdown("#### Harmonic Spectrum — FFT Analysis")

    thd_vals = {}
    fig_fft = go.Figure()
    colors_fft = ["#2196F3", "#FF5722", "#FFC107"]
    fill_colors = ["rgba(33,150,243,0.08)", "rgba(255,87,34,0.08)", "rgba(255,193,7,0.08)"]

    for i, (label, (col_name, _)) in enumerate(sig_map.items()):
        sig = df[col_name].values
        thd, xf, yf = compute_thd(sig, fs)
        thd_vals[label] = thd
        f_mask = xf <= 2000
        fig_fft.add_trace(go.Scatter(
            x=xf[f_mask], y=yf[f_mask],
            name=label, line=dict(color=colors_fft[i], width=1.5),
            fill="tozeroy", fillcolor=fill_colors[i],
            mode="lines",
        ))

    fig_fft.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        legend=dict(orientation="h", y=1.08),
        height=380,
        margin=dict(t=40, b=40),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig_fft, use_container_width=True)

    # ── THD Metrics ──────────────────────────
    st.markdown("#### Total Harmonic Distortion (THD)")
    c1, c2, c3 = st.columns(3)
    thd_cols = [c1, c2, c3]
    ieee_limit = 5.0
    for i, (label, thd) in enumerate(thd_vals.items()):
        status = "✅ Within IEEE 519" if thd < ieee_limit else "⚠️ Exceeds IEEE 519"
        color_badge = "#2ea043" if thd < ieee_limit else "#f85149"
        thd_cols[i].markdown(f"""
        <div class="metric-card" style="border-left-color:{color_badge};">
            <h3>{label}</h3>
            <p>{thd:.2f}%</p>
            <span style="color:{color_badge};font-size:0.82rem;">{status} (limit: 5%)</span>
        </div>""", unsafe_allow_html=True)

    # ── Per-phase subplots ───────────────────
    st.markdown("#### Individual Phase Signals")
    fig_sub = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=["Phase A — Signal 1", "Phase B — Signal 2", "Phase C — Signal 3"],
        vertical_spacing=0.08,
    )
    for i, (label, (col_name, color)) in enumerate(sig_map.items()):
        fig_sub.add_trace(
            go.Scatter(x=dff["Time"], y=dff[col_name],
                       line=dict(color=color, width=1),
                       name=label, showlegend=False),
            row=i + 1, col=1,
        )
    fig_sub.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=560,
        margin=dict(t=50, b=40),
    )
    fig_sub.update_xaxes(gridcolor="#21262d")
    fig_sub.update_yaxes(gridcolor="#21262d")
    fig_sub.update_annotations(font_color="#8b949e")
    st.plotly_chart(fig_sub, use_container_width=True)

    # ── Working Principle explanation ────────
    st.markdown("### How It Works")
    st.markdown("""
    <div class="info-box">
        <h4>PWM Switching & Harmonic Generation</h4>
        <p>
        The simulation operates at a <strong>discrete timestep of 50 µs</strong> (20 kHz sampling).
        The three-phase source feeds into a <strong>2-level PWM inverter</strong>,
        which chops the DC bus voltage using IGBTs at the switching frequency. This
        produces a three-phase output that approximates a sinusoid but contains:
        </p>
        <ul>
            <li><strong>Fundamental (50 Hz):</strong> the desired output component</li>
            <li><strong>Switching harmonics:</strong> at multiples of the PWM carrier frequency</li>
            <li><strong>Intermodulation harmonics:</strong> sideband frequencies around the carrier</li>
            <li><strong>Low-order harmonics:</strong> 3rd, 5th, 7th due to nonlinear load and deadtime</li>
        </ul>
        <p>
        The envelope of the waveform shows a <strong>decaying oscillation</strong> settling
        from startup transient to steady state — visible in the simulation data where the
        peak amplitude decreases from ~1.2 to ~0.75 over the 0.2 s window, indicating
        the RLC load's transient response.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 6 — ADVANTAGES
# ═══════════════════════════════════════════════════════════════
elif page == "advantages":
    st.markdown("""
    <div class="section-header">
        <h2>✅ Advantages of Solid-State Transformers</h2>
    </div>
    """, unsafe_allow_html=True)

    advantages = [
        ("⚡ Compact Size & Lightweight",
         "High-frequency operation (kHz range) drastically reduces core and winding dimensions. "
         "SSTs can be 10× smaller and lighter than equivalent conventional transformers — "
         "critical for traction, offshore, and space-constrained applications."),
        ("🔄 Bidirectional Power Flow",
         "Unlike conventional transformers, SSTs support bidirectional energy flow, enabling "
         "Vehicle-to-Grid (V2G), battery storage integration, and regenerative braking in "
         "electric rail systems without additional converter stages."),
        ("🎯 Active Power Quality Control",
         "SSTs can actively compensate for harmonics, reactive power, and voltage unbalance "
         "in real time using their output-side inverter — functions that conventional "
         "transformers simply cannot perform."),
        ("🔒 Galvanic Isolation at High Frequency",
         "The high-frequency isolation transformer provides safe electrical separation "
         "between primary and secondary sides with much smaller magnetic components, "
         "meeting safety standards while reducing weight."),
        ("⚙️ Voltage Regulation & Sag Compensation",
         "The output voltage can be precisely regulated regardless of input variation. "
         "SSTs inherently compensate for grid voltage sags and swells, protecting "
         "sensitive industrial equipment without external UPS systems."),
        ("🌿 Renewable Energy Integration",
         "SSTs natively interface with DC microgrids, solar PV, and battery storage "
         "systems through their internal DC bus — greatly simplifying the architecture "
         "of smart grids and EV charging infrastructure."),
        ("📊 Smart Grid Capability",
         "Built-in communication interfaces and controllable power electronics allow "
         "SSTs to participate in demand response, frequency regulation, and grid "
         "diagnostics — transforming passive distribution infrastructure into active nodes."),
        ("🛡️ Fault Isolation",
         "Semiconductor switches can interrupt fault currents within microseconds — "
         "far faster than mechanical breakers. SSTs prevent fault propagation across "
         "interconnected grids, improving system resilience."),
    ]

    col1, col2 = st.columns(2)
    for i, (title, desc) in enumerate(advantages):
        card = f"""
        <div class="adv-card">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>"""
        if i % 2 == 0:
            col1.markdown(card, unsafe_allow_html=True)
        else:
            col2.markdown(card, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Comparison Chart")

    metrics = ["Size/Weight", "Efficiency", "Power Quality Control", "Bidirectional", "Cost", "Reliability"]
    conventional = [2, 7, 1, 1, 9, 8]
    sst_scores    = [9, 8, 9, 9, 3, 6]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=conventional + [conventional[0]],
        theta=metrics + [metrics[0]],
        fill="toself",
        name="Conventional Transformer",
        line_color="#8b949e",
        fillcolor="rgba(139,148,158,0.2)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=sst_scores + [sst_scores[0]],
        theta=metrics + [metrics[0]],
        fill="toself",
        name="Solid-State Transformer",
        line_color="#1f6feb",
        fillcolor="rgba(31,111,235,0.2)",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 10], gridcolor="#30363d", color="#8b949e"),
            angularaxis=dict(gridcolor="#30363d", color="#c9d1d9"),
        ),
        paper_bgcolor="#0d1117",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.1),
        height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 7 — APPLICATIONS
# ═══════════════════════════════════════════════════════════════
elif page == "applications":
    st.markdown("""
    <div class="section-header">
        <h2>🏭 Applications of Solid-State Transformers</h2>
    </div>
    """, unsafe_allow_html=True)

    apps = [
        ("🚆 Electric Traction & Railways",
         "Modern high-speed rail systems (e.g., Siemens, ABB traction systems) use SSTs "
         "to replace bulky on-board transformers in locomotives and EMUs. The weight "
         "savings directly translate to improved energy efficiency and higher payload capacity."),
        ("🔋 EV Charging Infrastructure",
         "SSTs serve as the core of ultra-fast EV charging stations, providing direct "
         "DC output at 400V or 800V without a separate AC-DC stage. They support V2G "
         "bidirectional charging and can simultaneously serve multiple charge points."),
        ("☀️ Smart Grid & Renewable Integration",
         "SSTs act as the interface node in smart distribution grids, connecting solar "
         "farms, wind turbines, and battery storage to the utility grid with active "
         "power management, voltage support, and harmonic filtering."),
        ("🏭 Industrial Power Distribution",
         "In factories with large variable-frequency drives (VFDs) and arc furnaces, "
         "SSTs provide localized harmonic compensation and voltage regulation, protecting "
         "sensitive equipment from power quality issues."),
        ("🏙️ DC Microgrids",
         "SSTs naturally create a DC bus between their stages, making them ideal "
         "building blocks for DC microgrid architectures in data centers, campuses, "
         "and residential solar+storage communities."),
        ("⚓ Marine & Offshore",
         "Shipboard power systems and offshore oil platform use SSTs to reduce "
         "transformer weight and provide precise power management for critical loads "
         "in corrosive and space-limited marine environments."),
        ("🏥 Hospital & Data Center UPS",
         "The sub-cycle response time of SST-based UPS systems ensures zero-interruption "
         "power quality for medical equipment and servers — far superior to conventional "
         "static transfer switches."),
        ("🌐 Medium-Voltage Direct Current (MVDC) Grids",
         "SSTs are the key enabling technology for MVDC distribution, which reduces "
         "line losses by 30–40% compared to MVAC at the same power level — critical "
         "for future grid decarbonisation."),
    ]

    col1, col2 = st.columns(2)
    for i, (title, desc) in enumerate(apps):
        card = f"""
        <div class="app-card">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>"""
        if i % 2 == 0:
            col1.markdown(card, unsafe_allow_html=True)
        else:
            col2.markdown(card, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Market Adoption Timeline")

    timeline_data = {
        "Application":    ["Railway Traction", "EV Fast Charging", "Smart Grid", "DC Microgrids", "Industrial", "Marine"],
        "TRL Level":      [8, 7, 6, 5, 6, 5],
        "Market Readiness": [90, 75, 55, 45, 60, 40],
    }
    tl_df = pd.DataFrame(timeline_data)
    fig_bar = px.bar(
        tl_df, x="Market Readiness", y="Application",
        orientation="h",
        color="TRL Level",
        color_continuous_scale=["#1f6feb", "#8957e5", "#2ea043"],
        labels={"Market Readiness": "Market Readiness (%)"},
        template="plotly_dark",
    )
    fig_bar.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=340,
        margin=dict(t=20, b=40),
        xaxis=dict(gridcolor="#21262d", range=[0, 100]),
        yaxis=dict(gridcolor="#21262d"),
        coloraxis_colorbar=dict(title="TRL"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("TRL = Technology Readiness Level (1–9 scale, 9 = full commercial deployment)")
