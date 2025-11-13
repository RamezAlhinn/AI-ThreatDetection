"""
Streamlit UI for AI Threat Detection Agent - Simplified & Visual
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings, validate_config, is_mock_mode
from threat_agent import ThreatDetectionAgent
from storage import create_store
from metrics import QualityMetrics, evaluate_predictions

# Page configuration
st.set_page_config(
    page_title="AI Threat Detection Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_sample_logs() -> List[str]:
    """Load sample logs from file."""
    sample_path = "data/sample_logs.txt"
    if not os.path.exists(sample_path):
        return ["2024-01-15 10:23:41 INFO user=alice action=login ip=192.168.1.100 status=success"]
    with open(sample_path, 'r') as f:
        logs = [line.strip() for line in f if line.strip()]
    return logs

def render_overview_tab():
    """Simplified overview with key visuals."""
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🛡️ AI Threat Detection Agent")
        st.markdown("### LLM-Powered Cybersecurity PoC")
    with col2:
        mode = "🎭 Mock Mode" if is_mock_mode() else "🌐 Live API"
        st.metric("Status", mode, "Active")
    
    st.divider()
    
    # Core capabilities - visual cards
    st.subheader("🎯 Core Capabilities")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🔍 AI Framework Evaluation
        - Compare LLM APIs vs self-hosted
        - Mock mode for reliability
        - Easy to swap backends
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 PoC AI Agent
        - Real-time threat detection
        - Natural language explanations
        - Actionable recommendations
        """)
    
    with col3:
        st.markdown("""
        #### 📊 Quality Monitoring
        - Accuracy metrics
        - Confusion matrix
        - Human-in-the-loop workflow
        """)
    
    st.divider()
    
    # Simple architecture diagram
    st.subheader("🏗️ Architecture")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("""
        **📥 INPUT**
        - Security logs
        - User events
        - System alerts
        """)
    
    with col2:
        st.markdown("""
        **⚙️ PROCESSING**
        - LLM Analysis
        - Pattern Matching
        - Risk Scoring
        """)
    
    with col3:
        st.markdown("""
        **📤 OUTPUT**
        - Threat Level
        - Explanation
        - Recommended Action
        """)
    
    # Flow diagram
    st.markdown("""
    ```
    Log Entry  →  Threat Agent  →  LLM Client  →  Classification
                                       ↓
                                  Mock/Real API
    ```
    """)
    
    st.divider()
    
    # Key features
    st.subheader("✨ Why This Design?")
    
    features = {
        "🎭 Mock Mode": "Demo-ready without network/API",
        "🪶 Lightweight": "No PyTorch/TensorFlow needed",
        "🔧 Modular": "Swap components easily",
        "📈 Quality-First": "Built-in monitoring"
    }
    
    cols = st.columns(4)
    for idx, (feature, desc) in enumerate(features.items()):
        with cols[idx]:
            st.metric(feature, desc)

def render_demo_tab():
    """Clean, focused demo interface."""
    st.header("🚀 Live Threat Detection")
    
    # Initialize agent
    agent = ThreatDetectionAgent()
    sample_logs = load_sample_logs()
    
    # Simple input selection
    st.markdown("### Select Logs to Analyze")
    
    input_method = st.radio(
        "Input method:",
        ["📋 Example Logs", "✍️ Custom Logs"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if input_method == "📋 Example Logs":
        selected_indices = st.multiselect(
            "Choose logs:",
            range(len(sample_logs)),
            format_func=lambda i: f"Log {i+1}: {sample_logs[i][:70]}...",
            default=[0, 4, 7]
        )
        logs_to_analyze = [sample_logs[i] for i in selected_indices]
    else:
        custom_logs = st.text_area(
            "Paste logs (one per line):",
            height=150,
            placeholder="2024-01-15 10:23:41 INFO user=alice action=login ip=192.168.1.100 status=success"
        )
        logs_to_analyze = [line.strip() for line in custom_logs.split('\n') if line.strip()]
    
    # Analyze button
    if st.button("🔍 Analyze Logs", type="primary", disabled=len(logs_to_analyze) == 0, use_container_width=True):
        with st.spinner("🤖 AI analyzing..."):
            results = agent.analyze_logs_batch(logs_to_analyze)
            st.session_state['demo_results'] = results
    
    # Display results
    if 'demo_results' in st.session_state and st.session_state['demo_results']:
        results = st.session_state['demo_results']
        
        st.divider()
        st.markdown("### 📊 Results")
        
        # Summary metrics
        summary = agent.get_summary_stats(results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ Benign", summary['benign_count'], f"{summary['benign_pct']:.0f}%")
        with col2:
            st.metric("⚠️ Suspicious", summary['suspicious_count'], f"{summary['suspicious_pct']:.0f}%")
        with col3:
            st.metric("🚨 Malicious", summary['malicious_count'], f"{summary['malicious_pct']:.0f}%")
        with col4:
            st.metric("🎯 Confidence", f"{summary['avg_confidence']:.0%}")
        
        # Results table with color coding
        st.markdown("### 📋 Detailed Analysis")
        
        for idx, result in enumerate(results):
            prediction = result['prediction']
            
            # Color based on threat level
            if prediction == 'malicious':
                color = "🚨"
                bg_color = "#ffebee"
            elif prediction == 'suspicious':
                color = "⚠️"
                bg_color = "#fff9e6"
            else:
                color = "✅"
                bg_color = "#e8f5e9"
            
            with st.container():
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                    <h4>{color} {prediction.upper()} (Confidence: {result['confidence']:.0%})</h4>
                    <p><strong>Log:</strong> {result['log']}</p>
                    <p><strong>Reason:</strong> {result['explanation']}</p>
                    <p><strong>Action:</strong> {result['recommended_action']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Save option
        st.divider()
        if st.button("💾 Save Results", use_container_width=True):
            store = create_store()
            ids = store.save_predictions_batch(results)
            st.success(f"✅ Saved {len(ids)} predictions")

def render_quality_tab():
    """Simplified quality monitoring."""
    st.header("📈 Quality Monitoring")
    
    st.markdown("### Evaluate AI performance against labeled data")
    
    try:
        store = create_store()
        labeled_df = store.load_labeled_data()
        
        st.info(f"📁 {len(labeled_df)} labeled examples loaded")
        
        if st.button("🔄 Run Evaluation", type="primary", use_container_width=True):
            with st.spinner("🤖 Evaluating..."):
                agent = ThreatDetectionAgent()
                
                predictions = []
                for _, row in labeled_df.iterrows():
                    result = agent.analyze_log(row['log'])
                    predictions.append(result)
                
                labeled_df['prediction'] = [p['prediction'] for p in predictions]
                labeled_df['confidence'] = [p['confidence'] for p in predictions]
                labeled_df['explanation'] = [p['explanation'] for p in predictions]
                
                st.session_state['evaluation_df'] = labeled_df
        
        if 'evaluation_df' in st.session_state:
            eval_df = st.session_state['evaluation_df']
            metrics = QualityMetrics()
            summary, mistakes = evaluate_predictions(eval_df)
            
            st.divider()
            
            # Big metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Accuracy", f"{summary['overall_accuracy']:.0%}", help="Overall correct predictions")
            with col2:
                st.metric("📊 Total Samples", summary['total_samples'])
            with col3:
                errors = len(mistakes['all_errors'])
                st.metric("❌ Errors", errors, f"-{(errors/summary['total_samples']*100):.0f}%")
            
            st.divider()
            
            # Per-class metrics
            st.markdown("### 📊 Performance by Class")
            
            per_class_data = []
            for cls, metrics_dict in summary['per_class_metrics'].items():
                per_class_data.append({
                    'Threat Type': cls.capitalize(),
                    'Precision': f"{metrics_dict['precision']:.0%}",
                    'Recall': f"{metrics_dict['recall']:.0%}",
                    'F1-Score': f"{metrics_dict['f1']:.0%}",
                    'Count': metrics_dict['support']
                })
            
            st.dataframe(pd.DataFrame(per_class_data), use_container_width=True, hide_index=True)
            
            # Confusion matrix
            st.markdown("### 🔀 Confusion Matrix")
            st.caption("Rows = Actual | Columns = Predicted")
            
            confusion = summary['confusion_matrix']
            confusion_df = pd.DataFrame(confusion).T
            st.dataframe(confusion_df, use_container_width=True)
            
            # Key mistakes
            st.divider()
            st.markdown("### 🔍 Critical Mistakes")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fn_df = mistakes['false_negatives_malicious']
                st.markdown("#### 🚨 Missed Threats")
                if len(fn_df) > 0:
                    st.error(f"{len(fn_df)} malicious events missed!")
                    st.dataframe(fn_df[['log', 'true_label', 'prediction']], use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No missed threats")
            
            with col2:
                fp_df = mistakes['false_positives_malicious']
                st.markdown("#### ⚠️ False Alarms")
                if len(fp_df) > 0:
                    st.warning(f"{len(fp_df)} false alarms")
                    st.dataframe(fp_df[['log', 'true_label', 'prediction']], use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No false alarms")
    
    except FileNotFoundError:
        st.error("❌ Labeled data not found at `data/labeled_logs.csv`")

def render_framework_tab():
    """Simplified framework comparison."""
    st.header("🔧 Framework Evaluation")
    
    st.markdown("### Comparing AI Approaches for Security")
    
    # Comparison table
    comparison_data = {
        "Approach": [
            "🆓 Free LLM API",
            "🏠 Self-Hosted LLM",
            "💰 Commercial API",
            "📏 Rule-Based",
            "🌲 Classical ML"
        ],
        "Speed": ["Medium", "Fast", "Medium", "Very Fast", "Fast"],
        "Privacy": ["⚠️ External", "✅ Full", "⚠️ External", "✅ Local", "✅ Local"],
        "Cost": ["✅ Free", "💰 High", "💰 Per-use", "✅ Low", "✅ Low"],
        "Flexibility": ["High", "Very High", "High", "Low", "Medium"],
        "Explanations": ["✅ Yes", "✅ Yes", "✅ Yes", "❌ No", "❌ No"],
        "PoC Ready": ["✅ Yes", "⚠️ Setup", "✅ Yes", "✅ Yes", "✅ Yes"]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Why our choice
    st.markdown("### 🎯 Our Choice: Free LLM API + Mock Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Advantages**
        - No infrastructure needed
        - Free tier available
        - Natural language explanations
        - Mock mode = reliable demos
        - Fast to iterate
        """)
    
    with col2:
        st.markdown("""
        **⚠️ Trade-offs**
        - Network latency
        - Rate limits
        - Data privacy concerns
        - API dependency
        
        *Mitigated by mock mode!*
        """)

def render_future_tab():
    """Simplified future work."""
    st.header("🔮 Future Extensions")
    
    # Extension cards
    extensions = {
        "🤖 Multi-Agent System": [
            "Phishing email classifier",
            "SIEM alert triage",
            "Incident report drafter",
            "Vulnerability explainer"
        ],
        "👥 Human-in-the-Loop": [
            "Analyst feedback buttons",
            "Approval workflows",
            "Active learning",
            "Collaborative triage"
        ],
        "📊 Advanced Monitoring": [
            "Real-time dashboards",
            "Drift detection",
            "A/B testing",
            "Compliance reports"
        ],
        "🏢 Enterprise Integration": [
            "SIEM integration",
            "Ticketing systems",
            "Slack/Teams alerts",
            "Kubernetes deployment"
        ]
    }
    
    cols = st.columns(2)
    for idx, (title, features) in enumerate(extensions.items()):
        with cols[idx % 2]:
            st.markdown(f"### {title}")
            for feature in features:
                st.markdown(f"- {feature}")
            st.divider()
    
    # Job alignment
    st.markdown("### 🎓 Siemens Role Alignment")
    
    alignment = {
        "Evaluate AI Frameworks": "✅ Comparison methodology, abstraction layer",
        "Build PoC AI Agents": "✅ Working threat detection agent",
        "Monitor Output Quality": "✅ Metrics dashboard, evaluation workflow",
        "AI + Security Education": "✅ Domain-specific implementation",
        "Team Collaboration": "✅ Clean code, docs, extensibility"
    }
    
    for req, demo in alignment.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**{req}**")
        with col2:
            st.markdown(demo)

def main():
    """Main app."""
    
    # Sidebar
    with st.sidebar:
        st.title("🛡️ AI Threat Agent")
        st.caption("PoC for Siemens Interview")
        
        st.divider()
        
        # Status
        if is_mock_mode():
            st.success("🎭 Mock Mode Active")
            st.caption("Deterministic responses")
        else:
            st.info("🌐 API Mode Active")
            st.caption("Calling Hugging Face")
        
        st.divider()
        
        # Quick guide
        st.markdown("""
        **Quick Guide:**
        1. 📖 Overview
        2. 🚀 Try Live Demo
        3. 📈 Check Quality
        4. 🔧 See Comparison
        5. 🔮 Future Ideas
        """)
        
        st.divider()
        st.caption("Built with Streamlit")
        st.caption("Lightweight & Fast")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 Overview",
        "🚀 Live Demo",
        "📈 Quality",
        "🔧 Frameworks",
        "🔮 Future"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_demo_tab()
    
    with tab3:
        render_quality_tab()
    
    with tab4:
        render_framework_tab()
    
    with tab5:
        render_future_tab()

if __name__ == "__main__":
    main()
