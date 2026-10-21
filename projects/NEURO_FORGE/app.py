import json
import math
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components



# NEUROFORGE | C++17 Neural Microscope + Feature Influence Lab


ROOT = Path(__file__).resolve().parent
BRIDGE = ROOT / "neuroforge_bridge.exe"

FEATURES = [
    ("Orbital period", "Days", 365.25, 0.001, 100000.0, 36.525),
    ("Semi-major axis", "AU", 1.0, 0.001, 1000.0, 0.1),
    ("Orbital eccentricity", "Unitless", 0.02, 0.0, 1.0, 0.01),
    ("Equilibrium temperature", "K", 288.0, 1.0, 10000.0, 28.8),
    ("Planet radius", "Earth radii", 1.0, 0.01, 100.0, 0.1),
    ("Stellar temperature", "K", 5778.0, 500.0, 100000.0, 577.8),
    ("Stellar mass", "Solar masses", 1.0, 0.01, 100.0, 0.1),
    ("Stellar radius", "Solar radii", 1.0, 0.01, 1000.0, 0.1),
    ("Stellar metallicity", "dex", 0.0, -10.0, 10.0, 0.2),
]

st.set_page_config(
    page_title="NEUROFORGE | Neural Computation",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg:#050914;
    --panel:#0a1222;
    --line:#20344d;
    --cyan:#59e6ff;
    --muted:#91a8c1;
    --white:#e8f4ff;
}
html, body, [data-testid="stAppViewContainer"] {
    background:var(--bg);
    color:var(--white);
    font-family:'Space Grotesk',sans-serif;
}
[data-testid="stHeader"] { background:rgba(5,9,20,.95); }
[data-testid="stSidebar"] {
    background:#080e1b;
    border-right:1px solid var(--line);
}
.nf-eyebrow {
    font-family:'DM Mono',monospace;
    color:var(--cyan);
    font-size:11px;
    letter-spacing:2px;
    text-transform:uppercase;
}
.nf-title {
    font-size:clamp(34px,4vw,58px);
    font-weight:700;
    letter-spacing:-2px;
    line-height:1.05;
    margin:8px 0;
}
.nf-subtitle {
    color:var(--muted);
    font-size:14px;
    line-height:1.7;
    max-width:800px;
}
.nf-panel {
    background:linear-gradient(135deg,#0c1729,#080e1b);
    border:1px solid var(--line);
    border-radius:10px;
    padding:15px;
    min-height:88px;
    overflow-wrap:anywhere;
}
.nf-label {
    font-family:'DM Mono',monospace;
    color:#8ba2bc;
    font-size:10px;
    letter-spacing:1px;
}
.nf-value {
    font-family:'DM Mono',monospace;
    color:var(--cyan);
    font-size:19px;
    margin-top:9px;
}
.nf-rule {
    border:0;
    border-top:1px solid #1a2b43;
    margin:24px 0;
}
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stDownloadButton"] > button {
    width:100%;
    border-radius:7px;
    font-family:'DM Mono',monospace;
}
div[data-testid="stFormSubmitButton"] > button {
    background:var(--cyan);
    color:#04101c;
    border:1px solid var(--cyan);
    min-height:46px;
}
[data-testid="stMetric"] {
    background:#0a1222;
    border:1px solid var(--line);
    border-radius:9px;
    padding:12px;
}
[data-testid="stMetricValue"] {
    color:var(--cyan);
    font-family:'DM Mono',monospace;
}
[data-testid="stExpander"] {
    background:#080e1b;
    border:1px solid var(--line);
    border-radius:8px;
}
</style>
""", unsafe_allow_html=True)


# C++ BRIDGE 

def run_cpp(values):
    """Run local inference and return the bridge's JSON trace."""

    if not BRIDGE.is_file():
        raise FileNotFoundError(
            f"Cannot find {BRIDGE.name}.\nExpected location: {ROOT}"
        )

    if len(values) != 9:
        raise ValueError(
            f"Expected 9 input features, received {len(values)}."
        )

    command = [
        str(BRIDGE),
        *[format(float(value), ".12g") for value in values],
    ]

    result = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or f"Bridge exited with code {result.returncode}"
        )

    try:
        data = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The C++ bridge returned invalid JSON:\n"
            + result.stdout[:2000]
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            "The C++ bridge response must be a JSON object."
        )

    
    if data.get("status") not in ("ok", "success"):
        raise RuntimeError(f"Bridge reported an error: {data}")

    required = [
        "raw_inputs",
        "normalized_inputs",
        "feature_names",
        "input_hidden_weights",
        "hidden_biases",
        "hidden_weighted_sums",
        "hidden_activations",
        "hidden_output_weights",
        "output_bias",
        "output_weighted_sum",
        "prediction_score",
        "predicted_class",
        "predicted_label",
    ]

    missing = [key for key in required if key not in data]

    if missing:
        raise RuntimeError(
            f"Bridge JSON is missing fields: {missing}"
        )

    try:
        if (
            len(data["raw_inputs"]) != 9
            or len(data["normalized_inputs"]) != 9
            or len(data["feature_names"]) != 9
            or len(data["input_hidden_weights"]) != 8
            or any(
                len(row) != 9
                for row in data["input_hidden_weights"]
            )
            or len(data["hidden_biases"]) != 8
            or len(data["hidden_weighted_sums"]) != 8
            or len(data["hidden_activations"]) != 8
            or len(data["hidden_output_weights"]) != 8
        ):
            raise ValueError(
                "The bridge returned an unexpected network shape."
            )

        score = float(data["prediction_score"])
        predicted_class = int(data["predicted_class"])

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Invalid prediction score: {score}"
            )

        if predicted_class not in (0, 1):
            raise ValueError(
                f"Invalid predicted class: {predicted_class}"
            )

    except (TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"The bridge returned invalid trace data: {exc}"
        ) from exc

    return data


# SESSION STATE 

SESSION_DEFAULTS = {
    "nf_trace": None,
    "nf_raw_values": None,
    "nf_error": None,
    "nf_sweep": None,
    "nf_sweep_feature": None,
    "nf_eval_results": [],
    "nf_eval_errors": [],
    "nf_eval_requested": None,
}

for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# SIDEBAR 

with st.sidebar:
    st.markdown(
        '<div class="nf-eyebrow">✦ Mission configuration</div>',
        unsafe_allow_html=True,
    )
    st.markdown("## NEUROFORGE")
    st.caption("C++17 · From-scratch neural network")
    st.markdown("---")

    st.markdown("**NETWORK ARCHITECTURE**")
    st.code("9 inputs → 8 hidden → 1 output")

    st.markdown("**MODEL TARGET**")
    st.caption(
        "Transit-method class vs. non-transit-method class"
    )
    st.markdown("---")
    st.markdown("### Input parameters")
    st.caption(
        "Set all nine features, then run local inference."
    )

    with st.form("neuroforge_inputs"):
        raw_values = []

        for i, (
            name, unit, default, minimum, maximum, _
        ) in enumerate(FEATURES):
            raw_values.append(
                float(
                    st.number_input(
                        f"{i + 1:02d} · {name} ({unit})",
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=float(default),
                        format="%.6f",
                        key=f"nf_feature_{i}",
                    )
                )
            )

        submitted = st.form_submit_button(
            "▶ PROCESS THROUGH NETWORK",
            width="stretch",
        )

    st.markdown("---")
    st.markdown("**ENGINE STATUS**")
    st.caption(
        "Bridge: "
        + ("AVAILABLE" if BRIDGE.is_file() else "NOT FOUND")
    )
    st.caption("Inference: local C++ executable")
    st.caption("External AI API: not required")


# RUN BASELINE INFERENCE 

if submitted:
    try:
        with st.spinner("Running the C++ neural network..."):
            trace_result = run_cpp(raw_values)

        # Save the full successful response for every dashboard section.
        st.session_state["nf_trace"] = trace_result
        st.session_state["nf_raw_values"] = list(raw_values)
        st.session_state["nf_error"] = None
        st.session_state["nf_sweep"] = None
        st.session_state["nf_sweep_feature"] = None

    except Exception as exc:
        st.session_state["nf_error"] = str(exc)

if st.session_state["nf_error"]:
    st.error(st.session_state["nf_error"])


# HEADER 

st.markdown(
    '<div class="nf-eyebrow">'
    'Scientific machine learning / local inference'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="nf-title">'
    'NEUROFORGE<span style="color:#59e6ff">.</span>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="nf-subtitle">'
    'An interactive neural computation laboratory. Inspect the actual '
    'weights and activations returned by your C++ model, then test how '
    'its output responds when one input feature changes.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown('<hr class="nf-rule">', unsafe_allow_html=True)


# WAITING SCREEN

trace = st.session_state["nf_trace"]

if trace is None:
    st.markdown("""
    <div style="
        border:1px solid #20344d;border-radius:12px;padding:42px 22px;
        background:radial-gradient(ellipse at center,#10243a 0%,#080e1b 70%);
        text-align:center;">
      <div style="font-family:monospace;color:#59e6ff;letter-spacing:3px;font-size:11px;">
        NEURAL ENGINE / STANDBY
      </div>
      <div style="font-size:25px;font-weight:600;margin-top:18px;">
        Awaiting input signal
      </div>
      <div style="color:#91a8c1;line-height:1.8;max-width:570px;margin:12px auto;">
        Configure the nine features in the sidebar and process them through
        the local C++ executable. The microscope and influence lab will then
        use the returned model trace.
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # TRACE VALUES 

    normalized_inputs = trace["normalized_inputs"]
    input_hidden_weights = trace["input_hidden_weights"]
    hidden_biases = trace["hidden_biases"]
    hidden_sums = trace["hidden_weighted_sums"]
    hidden_activations = trace["hidden_activations"]
    hidden_output_weights = trace["hidden_output_weights"]
    output_bias = float(trace["output_bias"])
    output_sum = float(trace["output_weighted_sum"])
    prediction = float(trace["prediction_score"])
    predicted_class = int(trace["predicted_class"])
    predicted_label = str(trace["predicted_label"])

    a, b, c, d = st.columns(4)

    with a:
        st.markdown(
            '<div class="nf-panel">'
            '<div class="nf-label">ARCHITECTURE</div>'
            '<div class="nf-value">9 → 8 → 1</div></div>',
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            '<div class="nf-panel">'
            '<div class="nf-label">ACTIVATION</div>'
            '<div class="nf-value">Sigmoid</div></div>',
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            '<div class="nf-panel">'
            '<div class="nf-label">OUTPUT SCORE</div>'
            f'<div class="nf-value">{prediction:.6f}</div></div>',
            unsafe_allow_html=True,
        )

    with d:
        st.markdown(
            '<div class="nf-panel">'
            '<div class="nf-label">PREDICTED CLASS</div>'
            f'<div class="nf-value" style="font-size:14px">'
            f'{predicted_label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="nf-rule">', unsafe_allow_html=True)

  
    # NEURAL MICROSCOPE
   

    st.markdown(
        '<div class="nf-eyebrow">01 / Neural microscope</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Explore the network")
    st.caption(
        "The animation uses the forward-pass values returned by the C++ "
        "bridge. Click a hidden neuron to inspect its actual values."
    )

    visual_data = {
        "normalizedInputs": normalized_inputs,
        "inputHiddenWeights": input_hidden_weights,
        "hiddenBiases": hidden_biases,
        "hiddenSums": hidden_sums,
        "hiddenActivations": hidden_activations,
        "hiddenOutputWeights": hidden_output_weights,
        "outputBias": output_bias,
        "outputSum": output_sum,
        "prediction": prediction,
        "predictedClass": predicted_class,
        "predictedLabel": predicted_label,
        "rawInputs": trace["raw_inputs"],
        "featureNames": trace["feature_names"],
    }

    payload = json.dumps(
        visual_data,
        allow_nan=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")

    html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
*{box-sizing:border-box}
html,body{
 margin:0;padding:0;background:#050914;color:#e8f4ff;
 font-family:Consolas,monospace
}
.wrap{
 border:1px solid #1b304a;border-radius:12px;overflow:hidden;
 background:radial-gradient(ellipse at 48% 45%,rgba(27,70,108,.18),transparent 65%),#050914
}
.topline{
 display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;
 padding:12px 15px;border-bottom:1px solid #172a42;
 color:#88a3c0;font-size:10px;letter-spacing:1.2px
}
#stage{color:#59e6ff}
canvas{display:block;width:100%;height:500px}
.legend{
 display:flex;gap:13px;flex-wrap:wrap;padding:10px 15px;
 border-top:1px solid #172a42;color:#8ca2bb;font-size:10px
}
.dot{
 display:inline-block;width:7px;height:7px;border-radius:50%;
 margin-right:5px
}
.detail{
 margin-top:10px;border:1px solid #20344d;border-radius:10px;
 background:#080f1e;padding:15px
}
.heading{color:#59e6ff;font-size:11px;letter-spacing:1.3px;margin-bottom:10px}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.cell{background:#0b1728;border:1px solid #1a2b43;border-radius:6px;padding:9px;min-width:0}
.label{color:#7f96b1;font-size:9px;margin-bottom:7px}
.value{color:#e5f7ff;font-size:12px;overflow-wrap:anywhere}
.note{color:#91a8c1;font-size:10px;line-height:1.8;margin-top:12px;overflow-wrap:anywhere}
@media(max-width:650px){
 canvas{height:440px}
 .grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}
</style>
</head>
<body>
<div class="wrap">
 <div class="topline">
  <span>NEURAL PROCESSING / RETURNED FORWARD-PASS TRACE</span>
  <span id="stage">TRACE LOADED</span>
 </div>
 <canvas id="network"></canvas>
 <div class="legend">
  <span><i class="dot" style="background:#59e6ff"></i>POSITIVE WEIGHT</span>
  <span><i class="dot" style="background:#ff6da9"></i>NEGATIVE WEIGHT</span>
  <span><i class="dot" style="background:#c4f8ff"></i>SELECTED NEURON</span>
  <span>CLICK A HIDDEN NEURON</span>
 </div>
</div>
<div class="detail">
 <div class="heading" id="detailTitle">HIDDEN NEURON INSPECTOR / H1</div>
 <div class="grid" id="details"></div>
 <div class="note" id="note">Select a hidden neuron to inspect its values.</div>
</div>
<script>
const D = __TRACE__;
const canvas = document.getElementById("network");
const ctx = canvas.getContext("2d");
const stageEl = document.getElementById("stage");
const details = document.getElementById("details");
const detailTitle = document.getElementById("detailTitle");
let W=1000,H=500,selected=0;
const start=performance.now();

const inputNames=[
 "PERIOD","AXIS","ECCENTRICITY","EQ TEMP","PLANET RADIUS",
 "STAR TEMP","STAR MASS","STAR RADIUS","METALLICITY"
];

function fmt(v){
 const n=Number(v);
 return Number.isFinite(n)?n.toFixed(5):"N/A";
}
function resize(){
 const rect=canvas.getBoundingClientRect();
 const dpr=Math.min(window.devicePixelRatio||1,2);
 W=Math.max(320,rect.width);H=rect.height;
 canvas.width=Math.round(W*dpr);
 canvas.height=Math.round(H*dpr);
 ctx.setTransform(dpr,0,0,dpr,0,0);
}
window.addEventListener("resize",resize);
resize();

function positions(){
 const left=W<650?42:100;
 const mid=W*.51;
 const right=W-(W<650?45:100);
 const top=58,bottom=H-55;
 function ys(n,a,b){
  return Array.from({length:n},(_,i)=>a+(b-a)*i/(n-1));
 }
 return {
  input:ys(9,top,bottom).map(y=>({x:left,y})),
  hidden:ys(8,top+8,bottom-8).map(y=>({x:mid,y})),
  output:[{x:right,y:H/2}]
 };
}
function label(text,x,y){
 ctx.font="10px Consolas,monospace";
 ctx.textAlign="center";ctx.fillStyle="#7d9ab8";
 ctx.fillText(text,x,y);
}
function weight(i,h){
 return Number(D.inputHiddenWeights[h][i])||0;
}
function drawGrid(){
 ctx.strokeStyle="rgba(63,103,143,.10)";
 ctx.lineWidth=1;
 for(let x=0;x<W;x+=28){
  ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();
 }
 for(let y=0;y<H;y+=28){
  ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();
 }
}
function drawEdges(P,now,phase){
 for(let h=0;h<8;h++){
  for(let i=0;i<9;i++){
   const a=P.input[i],b=P.hidden[h],w=weight(i,h);
   const rgb=w>=0?"89,230,255":"255,109,169";
   ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
   ctx.lineWidth=.45+Math.min(Math.abs(w),2.5)*.5;
   ctx.strokeStyle=`rgba(${rgb},${phase>=1?.20:.08})`;
   ctx.stroke();
   if(phase===1){
    const t=(now*.00016+i*.117+h*.083)%1;
    ctx.beginPath();
    ctx.arc(a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t,1.8,0,Math.PI*2);
    ctx.fillStyle=`rgba(${rgb},.95)`;ctx.fill();
   }
  }
 }
 for(let h=0;h<8;h++){
  const a=P.hidden[h],b=P.output[0];
  const w=Number(D.hiddenOutputWeights[h])||0;
  const rgb=w>=0?"89,230,255":"255,109,169";
  ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
  ctx.lineWidth=.7+Math.min(Math.abs(w),2.5)*.7;
  ctx.strokeStyle=`rgba(${rgb},${phase>=2?.28:.10})`;
  ctx.stroke();
  if(phase===2){
   const t=(now*.00022+h*.16)%1;
   ctx.beginPath();
   ctx.arc(a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t,2,0,Math.PI*2);
   ctx.fillStyle=`rgba(${rgb},.95)`;ctx.fill();
  }
 }
}
function neuron(p,name,value,radius,color,isSelected=false){
 ctx.save();
 ctx.shadowBlur=isSelected?20:9;ctx.shadowColor=color;
 ctx.beginPath();
 ctx.arc(p.x,p.y,radius+(isSelected?2:0),0,Math.PI*2);
 ctx.fillStyle="#081426";ctx.fill();
 ctx.lineWidth=isSelected?2.2:1.2;ctx.strokeStyle=color;ctx.stroke();
 ctx.shadowBlur=0;
 ctx.beginPath();ctx.arc(p.x,p.y,2,0,Math.PI*2);
 ctx.fillStyle=color;ctx.fill();
 ctx.textAlign="center";ctx.font="10px Consolas,monospace";
 ctx.fillStyle="#e6f6ff";ctx.fillText(name,p.x,p.y+radius+13);
 ctx.font="9px Consolas,monospace";ctx.fillStyle=color;
 ctx.fillText(fmt(value),p.x,p.y+radius+25);
 ctx.restore();
}
function inspect(h){
 selected=Math.max(0,Math.min(7,h));
 const cells=[
  ["WEIGHTED SUM (z)",D.hiddenSums[selected]],
  ["SIGMOID ACTIVATION",D.hiddenActivations[selected]],
  ["HIDDEN BIAS",D.hiddenBiases[selected]],
  ["OUTPUT WEIGHT",D.hiddenOutputWeights[selected]]
 ];
 detailTitle.textContent="HIDDEN NEURON INSPECTOR / H"+(selected+1);
 details.replaceChildren();
 for(const [title,value] of cells){
  const cell=document.createElement("div");cell.className="cell";
  const lab=document.createElement("div");lab.className="label";lab.textContent=title;
  const val=document.createElement("div");val.className="value";val.textContent=fmt(value);
  cell.append(lab,val);details.append(cell);
 }
 document.getElementById("note").textContent=
  "Incoming weights for H"+(selected+1)+": "+
  inputNames.map((name,i)=>name+"="+fmt(weight(i,selected))).join(" · ")+
  ". Values are from the returned C++ model trace.";
}
canvas.addEventListener("click",e=>{
 const rect=canvas.getBoundingClientRect();
 const x=e.clientX-rect.left,y=e.clientY-rect.top;
 let best=-1,dist=Infinity;
 positions().hidden.forEach((p,i)=>{
  const d=Math.hypot(p.x-x,p.y-y);
  if(d<dist){dist=d;best=i;}
 });
 if(best>=0&&dist<32)inspect(best);
});
canvas.addEventListener("mousemove",e=>{
 const rect=canvas.getBoundingClientRect();
 const x=e.clientX-rect.left,y=e.clientY-rect.top;
 canvas.style.cursor=positions().hidden.some(p=>Math.hypot(p.x-x,p.y-y)<28)
  ?"pointer":"default";
});
function frame(now){
 ctx.clearRect(0,0,W,H);drawGrid();
 const cycle=((now-start)/1000)%9;
 const phase=cycle<2?0:(cycle<5.5?1:2);
 stageEl.textContent=phase===0?"PHASE 01 / INPUT SIGNALS":
  phase===1?"PHASE 02 / HIDDEN ACTIVATIONS":"PHASE 03 / OUTPUT";
 const P=positions();
 label("INPUT LAYER · 9",P.input[0].x,24);
 label("HIDDEN LAYER · 8",P.hidden[0].x,24);
 label("OUTPUT · 1",P.output[0].x,24);
 drawEdges(P,now,phase);
 P.input.forEach((p,i)=>neuron(p,"I"+(i+1),D.normalizedInputs[i],5.5,"#59e6ff"));
 P.hidden.forEach((p,h)=>neuron(
  p,"H"+(h+1),D.hiddenActivations[h],7.5,
  h===selected?"#c4f8ff":"#7cc8ff",h===selected
 ));
 const outColor=D.predictedClass===1?"#59e6ff":"#ffb86b";
 neuron(P.output[0],"Y",D.prediction,13,outColor);
 ctx.textAlign="center";ctx.font="10px Consolas,monospace";
 ctx.fillStyle="#8ca2bb";ctx.fillText("OUTPUT SCORE",P.output[0].x,H-27);
 ctx.fillStyle=outColor;ctx.fillText(fmt(D.prediction),P.output[0].x,H-13);
 requestAnimationFrame(frame);
}
inspect(0);
requestAnimationFrame(frame);
</script>
</body>
</html>
""".replace("__TRACE__", payload)

    components.html(html, height=730, scrolling=False)

    # ========================================================
    # 02 / FEATURE INFLUENCE LAB
    # ========================================================

    st.markdown('<hr class="nf-rule">', unsafe_allow_html=True)
    st.markdown(
        '<div class="nf-eyebrow">02 / Feature influence lab</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### What happens when one input changes?")
    st.write(
        "Choose a feature and sweep it around the current input. Each point "
        "runs the local C++ executable again; the other eight inputs stay fixed."
    )

    selected_feature = st.selectbox(
        "Feature to investigate",
        options=list(range(len(FEATURES))),
        format_func=lambda i: (
            f"{i + 1:02d} · {FEATURES[i][0]} ({FEATURES[i][1]})"
        ),
        key="nf_sweep_feature_select",
    )

    feature_name, unit, _, minimum, maximum, delta = FEATURES[selected_feature]
    baseline_values = list(st.session_state["nf_raw_values"])

    st.caption(
        f"Current value: {baseline_values[selected_feature]:.6g} {unit}. "
        f"The sweep tests up to four steps either side, with a step size "
        f"of {delta:g} {unit}, clipped to the configured range."
    )

    run_sweep = st.button(
        "▶ RUN FEATURE SENSITIVITY SWEEP",
        width="stretch",
        key="nf_run_sweep",
    )

    if run_sweep:
        try:
            points = []
            baseline = baseline_values[selected_feature]

            with st.spinner(
                "Running local C++ inference for each test point..."
            ):
                for offset in range(-4, 5):
                    tested_value = min(
                        maximum,
                        max(minimum, baseline + offset * delta),
                    )
                    test_values = list(baseline_values)
                    test_values[selected_feature] = tested_value

                    sweep_result = run_cpp(test_values)

                    points.append({
                        "Offset (steps)": offset,
                        f"{feature_name} ({unit})": tested_value,
                        "Output score": float(
                            sweep_result["prediction_score"]
                        ),
                        "Predicted class": str(
                            sweep_result["predicted_label"]
                        ),
                    })

            st.session_state["nf_sweep"] = points
            st.session_state["nf_sweep_feature"] = selected_feature

        except Exception as exc:
            st.error(f"Feature sweep failed: {exc}")

    sweep = st.session_state["nf_sweep"]

    if (
        sweep
        and st.session_state["nf_sweep_feature"] == selected_feature
    ):
        sweep_df = pd.DataFrame(sweep)

        st.markdown("#### Model response")
        chart_df = sweep_df.set_index("Offset (steps)")[["Output score"]]
        st.line_chart(chart_df, height=300)

        base_score = float(trace["prediction_score"])
        min_score = float(sweep_df["Output score"].min())
        max_score = float(sweep_df["Output score"].max())

        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline score", f"{base_score:.8f}")
        m2.metric("Lowest sweep score", f"{min_score:.8f}")
        m3.metric("Highest sweep score", f"{max_score:.8f}")

        st.caption(
            f"Feature tested: {feature_name}. Score range across this sweep: "
            f"{max_score - min_score:.8f}. This is local model sensitivity "
            "for the selected input range, not causal influence or a measure "
            "of scientific importance."
        )

        with st.expander("View every C++ inference in this sweep"):
            st.dataframe(
                sweep_df,
                width="stretch",
                hide_index=True,
            )

        st.download_button(
            "Download feature sweep (CSV)",
            data=sweep_df.to_csv(index=False).encode("utf-8"),
            file_name="neuroforge_feature_sweep.csv",
            mime="text/csv",
            width="stretch",
        )

    elif sweep:
        st.info(
            "Select the feature used in the previous sweep to view its results."
        )



# MODEL ACCURACY & CONFIDENCE LAB

st.markdown('<hr class="nf-rule">', unsafe_allow_html=True)
st.markdown(
    '<div class="nf-eyebrow">03 / Model accuracy & confidence lab</div>',
    unsafe_allow_html=True,
)
st.header("Model Accuracy & Confidence Lab")
st.caption(
    "Local C++ batch evaluation against labelled exoplanet data. "
    "Results are diagnostic until the original training split is verified."
)

EVAL_FEATURES = [
    "pl_orbper",
    "pl_orbsmax",
    "pl_orbeccen",
    "pl_eqt",
    "pl_rade",
    "st_teff",
    "st_mass",
    "st_rad",
    "st_met",
]

LABEL_COLUMN = "label"
DATA_PATH = ROOT / "data" / "exoplanets_clean.csv"

st.info(
    "Target: transit-method class (1) versus non-transit-method class (0). "
    "The sigmoid output is a score, not necessarily a calibrated probability."
)

if not DATA_PATH.is_file():
    st.error(f"Dataset not found: {DATA_PATH}")

elif not BRIDGE.is_file():
    st.error(
        f"C++ bridge not found: {BRIDGE}. "
        "Compile the C++ bridge before running evaluation."
    )

else:
    try:
        eval_df = pd.read_csv(DATA_PATH, low_memory=False)

        required = EVAL_FEATURES + [LABEL_COLUMN]
        missing = [
            column for column in required
            if column not in eval_df.columns
        ]

        if missing:
            st.error("Missing CSV columns: " + ", ".join(missing))

        else:
            for column in required:
                eval_df[column] = pd.to_numeric(
                    eval_df[column],
                    errors="coerce",
                )

            eval_df = eval_df.replace(
                [float("inf"), float("-inf")],
                float("nan"),
            ).dropna(subset=required)

            eval_df = eval_df[
                eval_df[LABEL_COLUMN].isin([0, 1])
            ].copy()

            eval_df[LABEL_COLUMN] = eval_df[LABEL_COLUMN].astype(int)
            eval_df["_source_row"] = eval_df.index

            st.write(f"Usable labelled rows: **{len(eval_df):,}**")
            st.write(
                "Class counts: "
                + str(
                    eval_df[LABEL_COLUMN]
                    .value_counts()
                    .sort_index()
                    .to_dict()
                )
            )

            fraction = st.select_slider(
                "Evaluation fraction per class",
                options=[0.10, 0.15, 0.20, 0.25, 0.30],
                value=0.20,
                format_func=lambda value: f"{value:.0%}",
                key="nf_eval_fraction",
            )

            st.caption(
                "Sampling is reproducible but does not prove that "
                "selected rows were excluded from model training."
            )

            run_evaluation = st.button(
                "Run Model Evaluation",
                type="primary",
                key="nf_run_accuracy_lab",
            )

            if run_evaluation:
                st.session_state["nf_eval_results"] = []
                st.session_state["nf_eval_errors"] = []
                st.session_state["nf_eval_requested"] = None

                try:
                    parts = []

                    for _, group in eval_df.groupby(LABEL_COLUMN):
                        if group.empty:
                            continue

                        count = max(
                            1,
                            round(len(group) * fraction),
                        )
                        count = min(count, len(group))

                        parts.append(
                            group.sample(
                                n=count,
                                random_state=42,
                            )
                        )

                    if not parts:
                        raise ValueError(
                            "No labelled rows are available for evaluation."
                        )

                    test_df = (
                        pd.concat(parts)
                        .sample(frac=1, random_state=42)
                        .reset_index(drop=True)
                    )

                    total_rows = len(test_df)

                    input_lines = []

                    for _, row in test_df.iterrows():
                        values = [
                            float(row[column])
                            for column in EVAL_FEATURES
                        ]

                        if not all(
                            math.isfinite(value)
                            for value in values
                        ):
                            raise ValueError(
                                "A sampled row contains a non-finite feature."
                            )

                        input_lines.append(
                            ",".join(
                                format(value, ".17g")
                                for value in values
                            )
                        )

                    payload = "\n".join(input_lines) + "\n"

                    with st.spinner(
                        f"Running C++ batch inference on {total_rows:,} rows..."
                    ):
                        process = subprocess.run(
                            [str(BRIDGE), "--batch"],
                            input=payload,
                            capture_output=True,
                            text=True,
                            cwd=str(ROOT),
                            timeout=180,
                            check=False,
                        )

                    if process.returncode != 0:
                        details = (
                            process.stderr.strip()
                            or process.stdout.strip()
                            or "No diagnostic output was returned."
                        )
                        raise RuntimeError(
                            f"C++ batch process failed "
                            f"(exit code {process.returncode}): "
                            f"{details[:3000]}"
                        )

                    output_lines = [
                        line.strip()
                        for line in process.stdout.splitlines()
                        if line.strip()
                    ]

                    batch_results = []
                    batch_errors = []

                    for index, (_, row) in enumerate(
                        test_df.iterrows()
                    ):
                        source_row = int(row["_source_row"])

                        if index >= len(output_lines):
                            batch_errors.append({
                                "source_csv_row": source_row,
                                "error": "No prediction line was returned.",
                            })
                            continue

                        try:
                            prediction_data = json.loads(
                                output_lines[index]
                            )

                            # Support either a direct prediction object
                            # or a response wrapped in a "data" object.
                            if isinstance(
                                prediction_data.get("data"), dict
                            ):
                                prediction_data = prediction_data["data"]

                            if prediction_data.get("status") not in (
                                None, "ok", "success"
                            ):
                                raise ValueError(
                                    f"Bridge error: {prediction_data}"
                                )

                            predicted = int(
                                prediction_data["predicted_class"]
                            )
                            score = float(
                                prediction_data["prediction_score"]
                            )
                            actual = int(row[LABEL_COLUMN])

                            if predicted not in (0, 1):
                                raise ValueError(
                                    f"Invalid predicted class: {predicted}"
                                )

                            if (
                                not math.isfinite(score)
                                or not 0.0 <= score <= 1.0
                            ):
                                raise ValueError(
                                    f"Invalid prediction score: {score}"
                                )

                            expected_class = int(score >= 0.5)

                            if predicted != expected_class:
                                raise ValueError(
                                    "Predicted class disagrees with "
                                    "the score at the 0.5 threshold."
                                )

                            batch_results.append({
                                "source_csv_row": source_row,
                                "actual": actual,
                                "predicted": predicted,
                                "score_class_1": score,
                                "predicted_class_score": (
                                    score
                                    if predicted == 1
                                    else 1.0 - score
                                ),
                                "correct": actual == predicted,
                            })

                        except Exception as exc:
                            batch_errors.append({
                                "source_csv_row": source_row,
                                "error": str(exc),
                            })

                    if len(output_lines) > total_rows:
                        batch_errors.append({
                            "source_csv_row": -1,
                            "error": (
                                f"Bridge returned "
                                f"{len(output_lines) - total_rows} "
                                "unexpected extra output line(s)."
                            ),
                        })

                    st.session_state["nf_eval_results"] = batch_results
                    st.session_state["nf_eval_errors"] = batch_errors
                    st.session_state["nf_eval_requested"] = total_rows
                    st.session_state["nf_eval_fraction_used"] = fraction

                except subprocess.TimeoutExpired:
                    st.session_state["nf_eval_results"] = []
                    st.session_state["nf_eval_errors"] = []
                    st.error(
                        "Batch evaluation exceeded 180 seconds. "
                        "Check the C++ bridge and model files."
                    )

                except Exception as exc:
                    st.session_state["nf_eval_results"] = []
                    st.session_state["nf_eval_errors"] = []
                    st.error(f"Evaluation failed: {exc}")

            results = st.session_state.get("nf_eval_results", [])
            errors = st.session_state.get("nf_eval_errors", [])

            if results:
                result_df = pd.DataFrame(results)

                actual = result_df["actual"].astype(int)
                predicted = result_df["predicted"].astype(int)
                scores = result_df["score_class_1"].astype(float)

                tp = int(((actual == 1) & (predicted == 1)).sum())
                tn = int(((actual == 0) & (predicted == 0)).sum())
                fp = int(((actual == 0) & (predicted == 1)).sum())
                fn = int(((actual == 1) & (predicted == 0)).sum())

                accuracy = float((actual == predicted).mean())
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                specificity = tn / (tn + fp) if tn + fp else 0.0

                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0
                )
                balanced_accuracy = (recall + specificity) / 2.0
                brier = float(((scores - actual) ** 2).mean())

                st.subheader("Evaluation Results")
                st.caption(
                    f"Successful predictions: {len(result_df):,} · "
                    f"Failed predictions: {len(errors):,} · "
                    f"Requested rows: "
                    f"{st.session_state.get('nf_eval_requested') or 0:,}"
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{accuracy:.2%}")
                c2.metric("Precision (class 1)", f"{precision:.2%}")
                c3.metric("Recall (class 1)", f"{recall:.2%}")

                c4, c5, c6 = st.columns(3)
                c4.metric("F1 score (class 1)", f"{f1:.2%}")
                c5.metric("Balanced accuracy", f"{balanced_accuracy:.2%}")
                c6.metric("Brier score", f"{brier:.4f}")

                st.caption(
                    "The Brier score is the mean squared error between "
                    "the class-1 score and binary label. Lower is better, "
                    "but it does not establish calibration."
                )

                st.subheader("Confusion Matrix")
                st.caption(
                    "Rows = actual class; columns = predicted class."
                )

                st.dataframe(
                    pd.DataFrame(
                        [[tn, fp], [fn, tp]],
                        index=["Actual 0", "Actual 1"],
                        columns=["Predicted 0", "Predicted 1"],
                    ),
                    use_container_width=True,
                )

                st.subheader("Prediction Score Distribution")
                st.caption(
                    "Raw sigmoid scores grouped by actual class; "
                    "these are not necessarily calibrated probabilities."
                )

                score_distribution = pd.DataFrame({
                    "Class 0 (actual)": (
                        scores[actual == 0].reset_index(drop=True)
                    ),
                    "Class 1 (actual)": (
                        scores[actual == 1].reset_index(drop=True)
                    ),
                })

                st.line_chart(
                    score_distribution,
                    y_label="Score for class 1",
                )

                st.subheader("Score Calibration Check")

                calibration = pd.DataFrame({
                    "score_class_1": scores,
                    "actual": actual,
                })

                calibration["score_bin"] = pd.cut(
                    calibration["score_class_1"],
                    bins=[
                        -0.000001, 0.2, 0.4, 0.6, 0.8, 1.000001
                    ],
                    labels=[
                        "0.0–0.2",
                        "0.2–0.4",
                        "0.4–0.6",
                        "0.6–0.8",
                        "0.8–1.0",
                    ],
                    include_lowest=True,
                )

                calibration_table = (
                    calibration.groupby(
                        "score_bin",
                        observed=False,
                    )
                    .agg(
                        examples=("actual", "size"),
                        mean_score=("score_class_1", "mean"),
                        observed_class_1_rate=("actual", "mean"),
                    )
                    .reset_index()
                )

                st.dataframe(
                    calibration_table,
                    use_container_width=True,
                )

                st.subheader("Inspect Individual Predictions")

                result_view = result_df.copy()
                result_view["actual_label"] = result_view["actual"].map({
                    0: "Non-transit-method class",
                    1: "Transit-method class",
                })
                result_view["predicted_label"] = result_view["predicted"].map({
                    0: "Non-transit-method class",
                    1: "Transit-method class",
                })

                result_view["score_class_1"] = (
                    result_view["score_class_1"].round(6)
                )
                result_view["predicted_class_score"] = (
                    result_view["predicted_class_score"].round(6)
                )

                st.dataframe(
                    result_view[
                        [
                            "source_csv_row",
                            "actual_label",
                            "predicted_label",
                            "score_class_1",
                            "predicted_class_score",
                            "correct",
                        ]
                    ],
                    use_container_width=True,
                    height=350,
                )

                st.download_button(
                    "Download Evaluation Results CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="neuroforge_evaluation.csv",
                    mime="text/csv",
                    key="nf_download_evaluation",
                )

                if errors:
                    with st.expander(
                        f"Inspect failed predictions ({len(errors)})"
                    ):
                        st.dataframe(
                            pd.DataFrame(errors),
                            use_container_width=True,
                        )

                st.warning(
                    "These are diagnostic metrics on a reproducible sample "
                    "of the local dataset. They are not verified unseen-test "
                    "metrics until the original training split is recovered "
                    "or an independent holdout is evaluated. The scores and "
                    "score-bin table do not prove calibration."
                )

            elif st.session_state.get("nf_eval_requested"):
                st.info(
                    "No valid predictions are available for the last "
                    "evaluation. Inspect the errors and verify the batch "
                    "bridge output."
                )

            else:
                st.info(
                    "The evaluation has not run yet. Click "
                    "'Run Model Evaluation' above to start it."
                )

    except Exception as exc:
        st.error(f"Could not load evaluation dataset: {exc}")



# COMPUTATION RESULTS


st.markdown('<hr class="nf-rule">', unsafe_allow_html=True)
st.markdown(
    '<div class="nf-eyebrow">04 / Computation results</div>',
    unsafe_allow_html=True,
)
st.markdown("### What the network returned")

trace = st.session_state.get("nf_trace")

if not isinstance(trace, dict) or not trace:
    st.info(
        "NEURAL ENGINE / STANDBY\n\n"
        "No successful inference trace is available yet. "
        "Configure all nine input features in the sidebar and run "
        "local inference."
    )

else:
    required_trace_fields = [
        "output_weighted_sum",
        "prediction_score",
        "output_bias",
        "predicted_class",
        "predicted_label",
        "raw_inputs",
        "normalized_inputs",
        "hidden_weighted_sums",
        "hidden_activations",
        "hidden_biases",
        "input_hidden_weights",
        "hidden_output_weights",
    ]

    missing_fields = [
        key for key in required_trace_fields if key not in trace
    ]

    if missing_fields:
        st.error(
            "The C++ bridge returned an incomplete trace. Missing fields: "
            + ", ".join(missing_fields)
        )
        st.caption(
            "Check the JSON keys returned by neuroforge_bridge.exe."
        )

    else:
        try:
            output_sum = float(trace["output_weighted_sum"])
            prediction = float(trace["prediction_score"])
            output_bias = float(trace["output_bias"])
            predicted_class = int(trace["predicted_class"])
            predicted_label = str(trace["predicted_label"])

            if not all(
                math.isfinite(value)
                for value in [output_sum, prediction, output_bias]
            ):
                raise ValueError(
                    "The C++ trace contains a non-finite numeric value."
                )

            if not 0.0 <= prediction <= 1.0:
                raise ValueError(
                    "Prediction score is outside the sigmoid range [0, 1]."
                )

            if predicted_class not in (0, 1):
                raise ValueError("Predicted class must be 0 or 1.")

            left, right = st.columns(2)

            with left:
                st.metric(
                    "Output Weighted Sum",
                    f"{output_sum:.8f}",
                )
                st.caption(
                    "Weighted input to the output sigmoid."
                )

            with right:
                st.metric(
                    "Model Output Score",
                    f"{prediction:.12f}",
                )
                st.caption(
                    "Raw sigmoid score for the model's training target, "
                    "not a calibrated probability that an object is "
                    "an exoplanet."
                )

            st.markdown(
                f"**Predicted label:** `{predicted_label}`"
            )
            st.markdown(
                f"**Predicted class:** `{predicted_class}`"
            )

            st.warning(
                "Scientific limitation: NeuroForge predicts transit-method "
                "class versus non-transit-method class from its training "
                "labels. It does not establish whether an object is a real "
                "exoplanet or confirm a new discovery."
            )

            with st.expander("Inspect the actual C++ computation trace"):
                st.markdown("**Raw Inputs**")
                st.json(trace["raw_inputs"])

                st.markdown("**Normalized Inputs**")
                st.json(trace["normalized_inputs"])

                st.markdown("**Hidden Weighted Sums**")
                st.json(trace["hidden_weighted_sums"])

                st.markdown("**Hidden Activations**")
                st.json(trace["hidden_activations"])

                st.markdown("**Hidden Biases**")
                st.json(trace["hidden_biases"])

                st.markdown("**Input-to-Hidden Weights (8 × 9)**")
                st.json(trace["input_hidden_weights"])

                st.markdown("**Hidden-to-Output Weights**")
                st.json(trace["hidden_output_weights"])

                st.markdown("**Output Calculation**")
                st.json({
                    "output_bias": output_bias,
                    "output_weighted_sum": output_sum,
                    "prediction_score": prediction,
                    "predicted_class": predicted_class,
                    "predicted_label": predicted_label,
                })

            trace_json = json.dumps(
                trace,
                indent=2,
                allow_nan=False,
            )

            st.download_button(
                "Download Actual Computation Trace (JSON)",
                data=trace_json,
                file_name="neuroforge_computation_trace.json",
                mime="application/json",
                key="nf_download_computation_trace",
            )

        except (TypeError, ValueError, KeyError) as exc:
            st.error(
                f"Could not display the computation trace: {exc}"
            )
            st.caption(
                "Run inference again and verify the output schema "
                "from neuroforge_bridge.exe."
            )




st.markdown('<hr class="nf-rule">', unsafe_allow_html=True)
st.caption(
    "NEUROFORGE · Local C++17 inference · Neural Microscope · "
    "Feature sensitivity · No external AI API"
)
