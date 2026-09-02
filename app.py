import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration & States ---
st.set_page_config(page_title="Clonal Evolution Simulator", layout="wide")

INITIAL_STATES = {
    'Normal Diploid (AB)': {'c': 2, 'b': 1},
    'Early CN-LOH (AA)': {'c': 2, 'b': 0},
    'Mono-allelic Deletion (A)': {'c': 1, 'b': 0},
    'Unbalanced Trisomy (AAB)': {'c': 3, 'b': 1},
    'Mono-allelic Trisomy (AAA)': {'c': 3, 'b': 0}
}

EVENTS = {
    'none': 'None (Maintained State)',
    'delA': 'Delete Major Allele',
    'delB': 'Delete Minor Allele',
    'gainA': 'Gain Major Allele',
    'gainB': 'Gain Minor Allele',
    'wgd': 'Whole-Genome Duplication',
    'submix': 'Subclonal Mix (+1 Minor Allele)'
}

# --- Admixture Math & Genotype Logic ---
def get_g2(base_g, event):
    c, b = base_g['c'], base_g['b']
    if event == 'none': return {'c': c, 'b': b}
    if event == 'delA': return {'c': max(c - 1, b), 'b': b} 
    if event == 'delB': return {'c': max(c - 1, 0), 'b': max(b - 1, 0)}
    if event == 'gainA': return {'c': c + 1, 'b': b}
    if event == 'gainB': return {'c': c + 1, 'b': b + 1}
    if event == 'wgd': return {'c': c * 2, 'b': b * 2}
    if event == 'submix': return {'c': c + 0.5, 'b': b + 0.5}

def get_genotype_string(tumor_state):
    c, b = tumor_state['c'], tumor_state['b']
    if c == 0:
        return "Null (-)", 0, 0
    
    a_count = c - b
    b_count = b
    
    if isinstance(c, float) and not c.is_integer():
        return f"Mix (Avg A:{a_count}, B:{b_count})", a_count, b_count
        
    a_count, b_count = int(a_count), int(b_count)
    genotype = ("A" * a_count) + ("B" * b_count)
    return genotype, a_count, b_count

def calc_admixture(tumor, purity):
    bulk_c = (purity * tumor['c']) + ((1 - purity) * 2)
    bulk_b = (purity * tumor['b']) + ((1 - purity) * 1)
    
    log2 = -5.0 if bulk_c == 0 else float(np.log2(bulk_c / 2))
    baf = 0.5 if bulk_c == 0 else bulk_b / bulk_c
    return log2, min(max(baf, 0), 1)

def generate_scatter(x_start, x_end, base_baf, count=250):
    x = np.random.uniform(x_start, x_end, count)
    noise = np.random.uniform(-0.02, 0.02, count)
    target_baf = np.where(np.random.rand(count) > 0.5, base_baf, 1 - base_baf)
    y = np.clip(target_baf + noise, 0, 1)
    return x, y

# --- UI Sidebar ---
with st.sidebar:
    st.header("Parameters")
    purity = st.slider("Tumor Purity", 0.1, 1.0, 0.80, step=0.05)
    
    st.subheader("Branch 1: Base State")
    b1_label = st.selectbox("Initial Tumor State", list(INITIAL_STATES.keys()), index=2) # Changed default to Deletion
    
    st.subheader("Branch 2: Settings")
    is_linked = st.checkbox("Link Branches (Sequential)", value=True)
    b2_base_label = st.selectbox("Branch 2: Base State", list(INITIAL_STATES.keys()), index=0, disabled=is_linked)
    b2_event_key = st.selectbox("Next Event", list(EVENTS.keys()), format_func=lambda x: EVENTS[x], index=1) # Changed default to delA

# --- Routing ---
g1 = INITIAL_STATES[b1_label]
base_g2 = g1 if is_linked else INITIAL_STATES[b2_base_label]
g2 = get_g2(base_g2, b2_event_key)

g1_str, g1_a, g1_b = get_genotype_string(g1)
g2_str, g2_a, g2_b = get_genotype_string(g2)

r1_log2, r1_baf = calc_admixture(g1, purity)
r2_log2, r2_baf = calc_admixture(g2, purity)

# --- Dashboard Layout ---
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Branch 1 Genotype", value=g1_str, delta=f"A: {g1_a} | B: {g1_b}", delta_color="off")
with col2:
    st.metric(label="Branch 2 Genotype (Deep Deletion)" if g2['c']==0 else "Branch 2 Genotype", 
              value=g2_str, delta=f"A: {g2_a} | B: {g2_b}", delta_color="off")

# --- Plotting ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

b1_x, b1_y = generate_scatter(0, 1, r1_baf)
b2_x, b2_y = generate_scatter(1, 2, r2_baf)

fig.add_trace(go.Scatter(x=b1_x, y=b1_y, mode='markers', name='Branch 1 BAF', 
                         marker=dict(size=4, color='rgba(214, 39, 40, 0.5)')), secondary_y=False)
fig.add_trace(go.Scatter(x=b2_x, y=b2_y, mode='markers', name='Branch 2 BAF', 
                         marker=dict(size=4, color='rgba(31, 119, 180, 0.5)')), secondary_y=False)

fig.add_trace(go.Scatter(x=[0, 0.95], y=[r1_log2, r1_log2], mode='lines', name='Branch 1 Log2', 
                         line=dict(color='#d62728', width=5)), secondary_y=True)
fig.add_trace(go.Scatter(x=[1.05, 2], y=[r2_log2, r2_log2], mode='lines', name='Branch 2 Log2', 
                         line=dict(color='#1f77b4', width=5)), secondary_y=True)

# Dynamic Log2 Axis Scaling (Ensures deep deletions stay in frame)
max_abs_log2 = max(abs(r1_log2), abs(r2_log2))
log2_bound = max(2.1, max_abs_log2 + 0.3) 

# Add Genotype Annotations to Plot
fig.add_annotation(x=0.475, y=log2_bound * 0.9, text=f"Clone 1: {g1_str}", showarrow=False, font=dict(size=16, color="#d62728"), yref="y2")
fig.add_annotation(x=1.525, y=log2_bound * 0.9, text=f"Clone 2: {g2_str}", showarrow=False, font=dict(size=16, color="#1f77b4"), yref="y2")

# Layout Formatting
fig.update_layout(
    xaxis=dict(title='Genomic Position', range=[0, 2], showticklabels=False, showgrid=False, zeroline=False),
    height=600,
    margin=dict(l=60, r=60, t=40, b=40),
    showlegend=False
)

fig.update_yaxes(title_text="B-Allele Frequency", range=[-0.05, 1.05], tickvals=[0, 0.25, 0.5, 0.75, 1.0], secondary_y=False)
fig.update_yaxes(title_text="Log2 Ratio", range=[-log2_bound, log2_bound], showgrid=True, gridcolor='#eee', zeroline=True, zerolinecolor='#999', secondary_y=True)

st.plotly_chart(fig, use_container_width=True)
