import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration & States ---
st.set_page_config(page_title="Multi-Branch Clonal Evolution Simulator", layout="wide")

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

COLORS = [
    '#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

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

# --- State Management ---
if 'num_branches' not in st.session_state:
    st.session_state.num_branches = 2

# --- UI Sidebar ---
with st.sidebar:
    st.header("Global Parameters")
    purity = st.slider("Tumor Purity", 0.1, 1.0, 0.80, step=0.05)
    
    st.divider()
    st.subheader("Manage Branches")
    
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("➕ Add Branch"):
        st.session_state.num_branches += 1
    if col_btn2.button("➖ Remove Branch") and st.session_state.num_branches > 1:
        st.session_state.num_branches -= 1

    st.divider()
    
    branch_states = []
    
    # Dynamically generate UI for N branches
    for i in range(st.session_state.num_branches):
        with st.expander(f"Branch {i+1} Configuration", expanded=(i < 2)):
            if i == 0:
                b_label = st.selectbox("Initial Tumor State", list(INITIAL_STATES.keys()), index=2, key=f"b{i}_init")
                current_g = INITIAL_STATES[b_label]
                branch_states.append(current_g)
            else:
                is_linked = st.checkbox(f"Sequential: Link to Branch {i}", value=True, key=f"b{i}_link")
                b_base_label = st.selectbox("Base State (if unlinked)", list(INITIAL_STATES.keys()), index=0, disabled=is_linked, key=f"b{i}_base")
                # Default to 'delA' for Branch 2, otherwise 'none'
                default_idx = 1 if i == 1 else 0
                b_event_key = st.selectbox("Evolutionary Event", list(EVENTS.keys()), format_func=lambda x: EVENTS[x], index=default_idx, key=f"b{i}_event")
                
                base_g = branch_states[i-1] if is_linked else INITIAL_STATES[b_base_label]
                current_g = get_g2(base_g, b_event_key)
                branch_states.append(current_g)

# --- Dashboard Layout: Metrics ---
cols = st.columns(st.session_state.num_branches)
calculated_data = []

for i, col in enumerate(cols):
    g = branch_states[i]
    g_str, g_a, g_b = get_genotype_string(g)
    log2, baf = calc_admixture(g, purity)
    
    calculated_data.append({'log2': log2, 'baf': baf, 'str': g_str})
    
    with col:
        # Highlight Deep Deletion context visually
        title = f"Branch {i+1}" + (" (Deep Del)" if g['c'] == 0 else "")
        st.metric(label=title, value=g_str, delta=f"A: {g_a} | B: {g_b}", delta_color="off")

# --- Plotting ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

max_abs_log2 = 0

for i in range(st.session_state.num_branches):
    data = calculated_data[i]
    color = COLORS[i % len(COLORS)]
    
    # Update max log2 for dynamic axis scaling
    max_abs_log2 = max(max_abs_log2, abs(data['log2']))
    
    # Calculate X positions for this branch
    x_start = i
    x_end = i + 1
    
    b_x, b_y = generate_scatter(x_start, x_end, data['baf'])
    
    # BAF Scatter
    fig.add_trace(go.Scatter(x=b_x, y=b_y, mode='markers', name=f'Branch {i+1} BAF', 
                             marker=dict(size=4, color=color, opacity=0.5)), secondary_y=False)
    
    # Log2 Segment (Slightly shortened to show visual separation between branches)
    fig.add_trace(go.Scatter(x=[x_start, x_end - 0.05], y=[data['log2'], data['log2']], 
                             mode='lines', name=f'Branch {i+1} Log2', 
                             line=dict(color=color, width=5)), secondary_y=True)

# Dynamic Log2 Axis Scaling
log2_bound = max(2.1, max_abs_log2 + 0.3) 

# Add Annotations
for i in range(st.session_state.num_branches):
    data = calculated_data[i]
    color = COLORS[i % len(COLORS)]
    fig.add_annotation(x=i + 0.475, y=log2_bound * 0.9, text=f"Clone {i+1}: {data['str']}", 
                       showarrow=False, font=dict(size=14, color=color), yref="y2")

# Layout Formatting
fig.update_layout(
    xaxis=dict(title='Genomic Position (Sequential Branches)', range=[0, st.session_state.num_branches], 
               showticklabels=False, showgrid=False, zeroline=False),
    height=600,
    margin=dict(l=60, r=60, t=40, b=40),
    showlegend=False
)

fig.update_yaxes(title_text="B-Allele Frequency", range=[-0.05, 1.05], tickvals=[0, 0.25, 0.5, 0.75, 1.0], secondary_y=False)
fig.update_yaxes(title_text="Log2 Ratio", range=[-log2_bound, log2_bound], showgrid=True, gridcolor='#eee', zeroline=True, zerolinecolor='#999', secondary_y=True)

st.plotly_chart(fig, use_container_width=True)
