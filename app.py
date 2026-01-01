#!/usr/bin/env python3
"""
🚀 GeneLab Predictor - Web Application
======================================
A Streamlit web app for predicting gene expression under spaceflight conditions.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    1. Push to GitHub
    2. Go to share.streamlit.io
    3. Connect repo and deploy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
import io
from typing import Dict, List, Tuple

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="GeneLab Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #00d4ff !important;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff006e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.3)); }
        to { filter: drop-shadow(0 0 30px rgba(123, 44, 191, 0.5)); }
    }
    
    .subtitle {
        font-family: 'JetBrains Mono', monospace;
        color: #888;
        text-align: center;
        font-size: 1.1rem;
        margin-top: -10px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e30, #2a2a40);
        border: 1px solid #333;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Pathway indicators */
    .pathway-up {
        color: #ff006e;
        font-weight: 600;
    }
    
    .pathway-down {
        color: #00d4ff;
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #12121a;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #333;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #7b2cbf, #00d4ff);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(123, 44, 191, 0.5);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Data tables */
    .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================

STRESS_PATHWAYS = {
    "dna_damage": {
        "genes": ["TP53", "ATM", "BRCA1", "GADD45A", "CDKN1A", "MDM2", "CHEK1", "CHEK2"],
        "color": "#ff006e",
        "description": "DNA repair and cell cycle arrest"
    },
    "oxidative_stress": {
        "genes": ["NFE2L2", "HMOX1", "SOD1", "SOD2", "CAT", "GPX1", "NQO1", "TXNRD1"],
        "color": "#ff8500",
        "description": "Antioxidant response"
    },
    "immune_response": {
        "genes": ["IL1B", "IL6", "TNF", "NFKB1", "STAT1", "IFNG", "CXCL8", "CCL2"],
        "color": "#00d4ff",
        "description": "Inflammatory signaling"
    },
    "muscle_atrophy": {
        "genes": ["FOXO1", "FOXO3", "MSTN", "FBXO32", "TRIM63", "MYOG", "CTSL"],
        "color": "#7b2cbf",
        "description": "Muscle wasting pathways"
    },
    "apoptosis": {
        "genes": ["BAX", "BCL2", "CASP3", "CASP9", "CYCS", "APAF1", "BID"],
        "color": "#e63946",
        "description": "Programmed cell death"
    },
    "autophagy": {
        "genes": ["ATG5", "BECN1", "SQSTM1", "MTOR", "ULK1", "ATG7"],
        "color": "#2a9d8f",
        "description": "Cellular recycling"
    },
    "hypoxia": {
        "genes": ["HIF1A", "VEGFA", "LDHA", "PGK1", "ENO1", "SLC2A1", "CA9"],
        "color": "#e9c46a",
        "description": "Low oxygen response"
    },
}

RANDOM_STATE = 42

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

@st.cache_data
def generate_demo_data(n_genes: int = 2000, n_samples: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate simulated spaceflight data"""
    np.random.seed(RANDOM_STATE)
    
    stress_genes = list(set(g for p in STRESS_PATHWAYS.values() for g in p["genes"]))
    other_genes = [f"GENE_{i:04d}" for i in range(n_genes - len(stress_genes))]
    all_genes = stress_genes + other_genes
    
    n_total = n_samples * 2
    base_expr = np.clip(np.random.normal(8, 3, size=(len(all_genes), n_total)), 0, 16)
    
    sample_names = [f"GC_{i+1}" for i in range(n_samples)] + [f"SF_{i+1}" for i in range(n_samples)]
    
    # Add spaceflight effects
    for i in range(len(stress_genes)):
        base_expr[i, n_samples:] += np.random.uniform(0.5, 2.0)
    
    for i in range(len(stress_genes), len(stress_genes) + 50):
        base_expr[i, n_samples:] -= np.random.uniform(0.3, 1.5)
    
    base_expr += np.random.normal(0, 0.5, base_expr.shape)
    base_expr = np.clip(base_expr, 0, 16)
    
    expression_df = pd.DataFrame(base_expr, index=all_genes, columns=sample_names)
    
    metadata_df = pd.DataFrame({
        "sample_id": sample_names,
        "condition": ["Ground_Control"] * n_samples + ["Spaceflight"] * n_samples,
        "organism": "Mus musculus",
        "tissue": "Muscle",
    })
    
    return expression_df, metadata_df


def calculate_pathway_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pathway activity scores"""
    scores = {}
    df_genes_upper = df.index.str.upper()
    
    for name, info in STRESS_PATHWAYS.items():
        genes_upper = [g.upper() for g in info["genes"]]
        matching_idx = [i for i, g in enumerate(df_genes_upper) if g in genes_upper]
        
        if len(matching_idx) >= 2:
            subset = df.iloc[matching_idx]
            z_scored = stats.zscore(subset.values, axis=1)
            scores[name] = np.nanmean(z_scored, axis=0)
    
    return pd.DataFrame(scores, index=df.columns)


def create_features(expression_df: pd.DataFrame, pathway_scores: pd.DataFrame, n_genes: int = 100) -> pd.DataFrame:
    """Create ML features"""
    pathway_features = pathway_scores.add_prefix("pathway_")
    
    gene_variance = expression_df.var(axis=1).sort_values(ascending=False)
    top_genes = gene_variance.head(n_genes).index.tolist()
    gene_features = expression_df.loc[top_genes].T.add_prefix("gene_")
    
    return pd.concat([pathway_features, gene_features], axis=1).fillna(0)


def train_and_evaluate(X: pd.DataFrame, y: np.ndarray) -> Dict:
    """Train model and return results"""
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    model.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'importance': importance_df
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_pathway_radar(pathway_scores: pd.DataFrame, metadata: pd.DataFrame) -> go.Figure:
    """Create radar chart of pathway activation"""
    
    pathways = list(pathway_scores.columns)
    
    gc_mask = metadata['condition'] == 'Ground_Control'
    sf_mask = metadata['condition'] == 'Spaceflight'
    
    gc_means = pathway_scores[gc_mask.values].mean().values.tolist()
    sf_means = pathway_scores[sf_mask.values].mean().values.tolist()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=gc_means + [gc_means[0]],
        theta=pathways + [pathways[0]],
        fill='toself',
        fillcolor='rgba(0, 212, 255, 0.2)',
        line=dict(color='#00d4ff', width=2),
        name='Ground Control'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=sf_means + [sf_means[0]],
        theta=pathways + [pathways[0]],
        fill='toself',
        fillcolor='rgba(255, 0, 110, 0.2)',
        line=dict(color='#ff006e', width=2),
        name='Spaceflight'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-1.5, 1.5], gridcolor='#333'),
            angularaxis=dict(gridcolor='#333'),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(x=0.5, y=-0.1, xanchor='center', orientation='h'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#888'),
        margin=dict(t=40, b=60)
    )
    
    return fig


def create_pathway_bars(pathway_scores: pd.DataFrame, metadata: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart of pathway changes"""
    
    gc_mask = metadata['condition'].values == 'Ground_Control'
    sf_mask = metadata['condition'].values == 'Spaceflight'
    
    pathways = list(pathway_scores.columns)
    gc_means = [pathway_scores[p].values[gc_mask].mean() for p in pathways]
    sf_means = [pathway_scores[p].values[sf_mask].mean() for p in pathways]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Ground Control',
        x=pathways,
        y=gc_means,
        marker_color='#00d4ff'
    ))
    
    fig.add_trace(go.Bar(
        name='Spaceflight',
        x=pathways,
        y=sf_means,
        marker_color='#ff006e'
    ))
    
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#888'),
        xaxis=dict(gridcolor='#222'),
        yaxis=dict(gridcolor='#222', title='Activity Score'),
        legend=dict(x=0.5, y=1.1, xanchor='center', orientation='h'),
        margin=dict(t=60)
    )
    
    return fig


def create_heatmap(pathway_scores: pd.DataFrame, metadata: pd.DataFrame) -> go.Figure:
    """Create heatmap of pathway scores"""
    
    sorted_idx = metadata.sort_values('condition').index
    sorted_scores = pathway_scores.iloc[sorted_idx]
    sorted_samples = metadata.iloc[sorted_idx]['sample_id'].values
    
    fig = go.Figure(data=go.Heatmap(
        z=sorted_scores.T.values,
        x=sorted_samples,
        y=list(sorted_scores.columns),
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title='Score')
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#888'),
        margin=dict(t=40)
    )
    
    return fig


def create_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create feature importance bar chart"""
    
    top = importance_df.head(top_n).iloc[::-1]
    
    colors = ['#ff006e' if f.startswith('pathway_') else '#00d4ff' for f in top['feature']]
    
    fig = go.Figure(go.Bar(
        x=top['importance'],
        y=top['feature'],
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#888'),
        xaxis=dict(gridcolor='#222', title='Importance'),
        yaxis=dict(gridcolor='#222'),
        margin=dict(l=150, t=40)
    )
    
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-title">🚀 GeneLab Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict gene expression changes under spaceflight conditions</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        data_source = st.radio(
            "Data Source",
            ["🎮 Demo Data", "📤 Upload Your Data"],
            index=0
        )
        
        st.markdown("---")
        
        st.markdown("### 🧬 Pathways Analyzed")
        for name, info in STRESS_PATHWAYS.items():
            with st.expander(f"**{name.replace('_', ' ').title()}**"):
                st.markdown(f"*{info['description']}*")
                st.markdown(f"Genes: `{', '.join(info['genes'][:4])}...`")
        
        st.markdown("---")
        st.markdown("### 📚 Resources")
        st.markdown("[NASA GeneLab](https://genelab.nasa.gov/)")
        st.markdown("[Documentation](https://genelab.nasa.gov/genelabAPIs)")
    
    # Main content
    if data_source == "🎮 Demo Data":
        st.info("🎮 **Demo Mode**: Using simulated spaceflight data. Upload your own data for real analysis.")
        expression_df, metadata_df = generate_demo_data()
    else:
        st.markdown("### 📤 Upload Your Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            expr_file = st.file_uploader(
                "Expression Matrix (CSV)",
                type=['csv'],
                help="Genes as rows, samples as columns"
            )
        
        with col2:
            meta_file = st.file_uploader(
                "Sample Metadata (CSV)",
                type=['csv'],
                help="Must have 'sample_id' and 'condition' columns"
            )
        
        if not expr_file or not meta_file:
            st.warning("Please upload both files to continue.")
            
            with st.expander("📋 Expected Data Format"):
                st.markdown("""
                **Expression Matrix:**
                ```
                gene,Sample_1,Sample_2,Sample_3
                TP53,8.2,7.9,8.5
                ATM,6.1,6.3,5.8
                ```
                
                **Metadata:**
                ```
                sample_id,condition,organism,tissue
                Sample_1,Ground_Control,Mus musculus,Muscle
                Sample_2,Spaceflight,Mus musculus,Muscle
                ```
                """)
            return
        
        try:
            expression_df = pd.read_csv(expr_file, index_col=0)
            metadata_df = pd.read_csv(meta_file)
            st.success(f"✅ Loaded {expression_df.shape[0]} genes × {expression_df.shape[1]} samples")
        except Exception as e:
            st.error(f"Error loading files: {e}")
            return
    
    # Run Analysis Button
    if st.button("🔬 Run Analysis", type="primary", use_container_width=True):
        
        with st.spinner("Analyzing..."):
            # Calculate pathway scores
            pathway_scores = calculate_pathway_scores(expression_df)
            
            # Create features
            features = create_features(expression_df, pathway_scores)
            
            # Labels
            y = (metadata_df['condition'] == 'Spaceflight').astype(int).values
            
            # Train model
            results = train_and_evaluate(features, y)
        
        st.success("✅ Analysis complete!")
        st.markdown("---")
        
        # Results Section
        st.markdown("## 📊 Results")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{expression_df.shape[0]:,}</div>
                <div class="metric-label">Genes</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(metadata_df)}</div>
                <div class="metric-label">Samples</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cv_pct = f"{results['cv_mean']*100:.0f}%"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{cv_pct}</div>
                <div class="metric-label">CV Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(pathway_scores.columns)}</div>
                <div class="metric-label">Pathways</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Pathway Analysis
        st.markdown("### 🧬 Pathway Activation")
        
        tab1, tab2, tab3 = st.tabs(["📊 Radar Chart", "📈 Bar Chart", "🗺️ Heatmap"])
        
        with tab1:
            fig = create_pathway_radar(pathway_scores, metadata_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = create_pathway_bars(pathway_scores, metadata_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = create_heatmap(pathway_scores, metadata_df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Pathway changes summary
        st.markdown("### 📈 Pathway Changes Summary")
        
        changes_data = []
        gc_mask = metadata_df['condition'].values == 'Ground_Control'
        sf_mask = metadata_df['condition'].values == 'Spaceflight'
        
        for pathway in pathway_scores.columns:
            gc_mean = pathway_scores[pathway].values[gc_mask].mean()
            sf_mean = pathway_scores[pathway].values[sf_mask].mean()
            change = sf_mean - gc_mean
            changes_data.append({
                'Pathway': pathway.replace('_', ' ').title(),
                'Ground Control': f"{gc_mean:.2f}",
                'Spaceflight': f"{sf_mean:.2f}",
                'Change': f"{'↑' if change > 0 else '↓'} {abs(change):.2f}",
                'Status': '🔴 Activated' if change > 0.3 else '🔵 Suppressed' if change < -0.3 else '⚪ Normal'
            })
        
        st.dataframe(pd.DataFrame(changes_data), hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Feature Importance
        st.markdown("### 🎯 Top Predictive Features")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = create_importance_chart(results['importance'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Top 10 Features:**")
            for i, row in results['importance'].head(10).iterrows():
                ftype = "🔴 Pathway" if row['feature'].startswith('pathway_') else "🔵 Gene"
                name = row['feature'].replace('pathway_', '').replace('gene_', '')
                st.markdown(f"{ftype} **{name}**: {row['importance']:.3f}")
        
        st.markdown("---")
        
        # Downloads
        st.markdown("### 📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = pathway_scores.to_csv()
            st.download_button(
                "📊 Pathway Scores",
                csv,
                "pathway_scores.csv",
                "text/csv"
            )
        
        with col2:
            csv = results['importance'].to_csv(index=False)
            st.download_button(
                "🎯 Feature Importance",
                csv,
                "feature_importance.csv",
                "text/csv"
            )
        
        with col3:
            # Generate report
            report = f"""
GENELAB PREDICTOR REPORT
========================

Data Summary:
- Genes: {expression_df.shape[0]}
- Samples: {len(metadata_df)}
- Pathways: {len(pathway_scores.columns)}

Model Performance:
- CV Accuracy: {results['cv_mean']:.1%} ± {results['cv_std']:.1%}

Top Features:
"""
            for i, row in results['importance'].head(10).iterrows():
                report += f"- {row['feature']}: {row['importance']:.4f}\n"
            
            st.download_button(
                "📄 Analysis Report",
                report,
                "analysis_report.txt",
                "text/plain"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Built with NASA GeneLab Open Data | 
        <a href="https://genelab.nasa.gov/" style="color: #00d4ff;">GeneLab</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
