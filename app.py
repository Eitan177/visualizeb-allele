import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration & States ---
st.set_page_config(page_title="Evolution Simulator", layout="wide")

INITIAL_STATES = {
    'Normal Diploid (AB)': {'c': 2, 'b': 1},
    'Early CN-LOH (AA)': {'c': 2, 'b': 0},
    'Mono-allelic Deletion (A)': {'c': 1, 'b': 0},
    'Balanced Trisomy (AAA)': {'c': 3, 'b': 1}
}

EVENTS = {
    'none': 'None (Maintained State)',
    'delB': 'Delete Minor Allele',
    'delA': 'Delete Major Allele',
    'gainA': 'Gain Major Allele',
    'gainB': 'Gain Minor Allele',
    'wgd': 'Whole-Genome Duplication',
    'submix': 'Subclonal Mix (+1 Minor Allele)'
}

# --- Admixture Math ---
def get_g2(base_g, event):
    c, b = base_g['c'], base_g['b']
    if event == 'none': return {'c': c, 'b': b}
    if event == 'delB': return {'c': max(c - 1, 0), 'b': max(b - 1, 0)}
    if event == 'gainA': return {'c': c + 1, 'b': b}
    if event == 'gainB': return {'c': c + 1, 'b': b + 1}
    if event == 'wgd': return {'c': c * 2, 'b': b * 2}
    if event == 'submix': return {'c': c + 0.5, 'b': b + 0.5}

def calc_admixture(tumor, purity):
    bulk_c = (purity * tumor['c']) + ((1 - purity) * 2)
    bulk_b = (purity * tumor['b']) + ((1 - purity) * 1)
    
    log2 = -2.0 if bulk_c == 0 else float(np.log2(bulk_c / 2))
    baf = 0.5 if bulk_c == 0 else bulk_b / bulk_c
    return log2, min(max(baf, 0), 1)

def generate_scatter(x_start, x_end, base_baf, count=250):
    x = np.random.uniform(x_start, x_end, count)
    noise = np.random.uniform(-0.02, 0.02, count)
    # Symmetrically distribute bands if unbalanced
    target_baf = np.where(np.random.rand(count) > 0.5, base_baf, 1 - base_baf)
    y = np.clip(target_baf + noise, 0, 1)
    return x, y

def get_g2(base_g, event):
    c, b = base_g['c'], base_g['b']
    if event == 'none': return {'c': c, 'b': b}
    if event == 'delA': return {'c': max(c - 1, b), 'b': b}  # <-- Add this line
    if event == 'delB': return {'c': max(c - 1, 0), 'b': max(b - 1, 0)}
    if event == 'gainA': return {'c': c + 1, 'b': b}
    if event == 'gainB': return {'c': c + 1, 'b': b + 1}
    if event == 'wgd': return {'c': c * 2, 'b': b * 2}
    if event == 'submix': return {'c': c + 0.5, 'b': b + 0.5}

# --- UI Sidebar ---
with st.sidebar:
    st.header("Parameters")
    purity = st.slider("Tumor Purity", 0.1, 1.0, 0.80, step=0.05)
    
    st.subheader("Branch 1: Base State")
    b1_label = st.selectbox("Initial Tumor State", list(INITIAL_STATES.keys()), index=1)
    
    st.subheader("Branch 2: Settings")
    is_linked = st.checkbox("Link Branches (Sequential)", value=True)
    b2_base_label = st.selectbox("Branch 2: Base State", list(INITIAL_STATES.keys()), index=0, disabled=is_linked)
    b2_event_key = st.selectbox("Next Event", list(EVENTS.keys()), format_func=lambda x: EVENTS[x], index=5)

# --- Routing ---
g1 = INITIAL_STATES[b1_label]
base_g2 = g1 if is_linked else INITIAL_STATES[b2_base_label]
g2 = get_g2(base_g2, b2_event_key)

r1_log2, r1_baf = calc_admixture(g1, purity)
r2_log2, r2_baf = calc_admixture(g2, purity)

# --- Plotting ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Generate Points
b1_x, b1_y = generate_scatter(0, 1, r1_baf)
b2_x, b2_y = generate_scatter(1, 2, r2_baf)

# Add BAF Traces (Left Axis)
fig.add_trace(go.Scatter(x=b1_x, y=b1_y, mode='markers', name='Branch 1 BAF', 
                         marker=dict(size=4, color='rgba(214, 39, 40, 0.5)')), secondary_y=False)
fig.add_trace(go.Scatter(x=b2_x, y=b2_y, mode='markers', name='Branch 2 BAF', 
                         marker=dict(size=4, color='rgba(31, 119, 180, 0.5)')), secondary_y=False)

# Add Log2 Segments (Right Axis)
fig.add_trace(go.Scatter(x=[0, 0.95], y=[r1_log2, r1_log2], mode='lines', name='Branch 1 Log2', 
                         line=dict(color='#d62728', width=5)), secondary_y=True)
fig.add_trace(go.Scatter(x=[1.05, 2], y=[r2_log2, r2_log2], mode='lines', name='Branch 2 Log2', 
                         line=dict(color='#1f77b4', width=5)), secondary_y=True)

# Layout Formatting
fig.update_layout(
    title='Coupled/Uncoupled Tumor Evolution Simulator',
    xaxis=dict(title='Genomic Position', range=[0, 2], showticklabels=False, showgrid=False, zeroline=False),
    height=650,
    margin=dict(l=60, r=60, t=60, b=60),
    showlegend=False
)

# Synchronize Axes (BAF 0.5 == Log2 0)
fig.update_yaxes(title_text="B-Allele Frequency", range=[-0.05, 1.05], tickvals=[0, 0.25, 0.5, 0.75, 1.0], secondary_y=False)
fig.update_yaxes(title_text="Log2 Ratio", range=[-2.1, 2.1], tickvals=[-2, -1, 0, 1, 2], showgrid=True, gridcolor='#eee', secondary_y=True)

st.plotly_chart(fig, use_container_width=True)
