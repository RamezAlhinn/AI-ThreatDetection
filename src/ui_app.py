"""
Streamlit UI for AI Threat Detection Agent.

This interactive web application serves as both a functional demo
and a presentation tool for the interview.

Tabs:
1. Overview & Story - Project introduction and architecture
2. Live Demo - Interactive threat detection
3. Quality Monitoring - Metrics dashboard
4. Framework Evaluation - Comparison of AI approaches
5. Future Work & Siemens Fit - Extensions and job alignment
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings, validate_config, is_mock_mode
from threat_agent import ThreatDetectionAgent
from storage import create_store
from metrics import QualityMetrics, evaluate_predictions


# Page configuration
st.set_page_config(
    page_title="AI Threat Detection Agent - PoC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_sample_logs() -> List[str]:
    """Load sample logs from file."""
    sample_path = "data/sample_logs.txt"
    if not os.path.exists(sample_path):
        return [
            "2024-01-15 10:23:41 INFO user=alice action=login ip=192.168.1.100 status=success",
            "2024-01-15 10:30:22 ERROR user=unknown action=sql_query query=\"SELECT * FROM users WHERE '1'='1'\" ip=198.51.100.42 status=blocked",
        ]
    
    with open(sample_path, 'r') as f:
        logs = [line.strip() for line in f if line.strip()]
    return logs


def render_overview_tab():
    """Render the Overview & Story tab."""
    st.header("🎯 Overview & Story")
    
    st.markdown("""
    ## Welcome to the AI Threat Detection Agent PoC
    
    This proof-of-concept demonstrates an **LLM-powered cybersecurity threat detection system** 
    specifically designed to showcase skills relevant to the **Siemens AI/Cybersecurity Working Student** position.
    
    ### 🎓 Project Purpose
    
    This PoC directly addresses the three core responsibilities from the job description:
    
    1. **Evaluating AI frameworks and tools for security tasks**  
       → Framework comparison methodology (see "Framework Evaluation Notes" tab)
    
    2. **Building PoC AI agents for specific security use cases**  
       → End-to-end threat detection agent with live demo
    
    3. **Monitoring output quality and continuous improvement**  
       → Metrics dashboard with human-in-the-loop evaluation workflow
    
    ---
    
    ### 🏗️ System Architecture
    
    ```
    ┌───────────────────────────────────────────────────────┐
    │                    Web UI (Streamlit)                  │
    │  - Interactive demo  - Quality dashboard  - Evaluation │
    └────────────────────────┬──────────────────────────────┘
                             │
                             ▼
    ┌────────────────────────────────────────────────────────┐
    │              Threat Detection Agent                     │
    │  - Orchestrates pipeline  - Structures outputs          │
    └───────┬────────────────────────────────┬───────────────┘
            │                                │
            ▼                                ▼
    ┌──────────────────┐          ┌─────────────────────┐
    │   LLM Client     │          │   Storage Layer     │
    │  ┌────────────┐  │          │  - CSV/SQLite       │
    │  │ Real Mode  │  │          │  - Predictions      │
    │  │ (HF API)   │  │          │  - Labels           │
    │  └────────────┘  │          └─────────────────────┘
    │  ┌────────────┐  │
    │  │ Mock Mode  │  │
    │  │ (Patterns) │  │
    │  └────────────┘  │
    └──────────────────┘
            │
            ▼
    ┌──────────────────────┐
    │  Hugging Face API    │
    │  (Free Inference)    │
    └──────────────────────┘
    ```
    
    ---
    
    ### 🔄 Data Flow Walkthrough
    
    1. **Input**: Security log entries (web server logs, auth logs, system events)
    
    2. **Processing**:
       - Log sent to Threat Detection Agent
       - Agent calls LLM Client (mock or real mode)
       - LLM analyzes log and returns structured classification
    
    3. **Output**:
       - **Prediction**: benign / suspicious / malicious
       - **Confidence**: 0.0 - 1.0 probability score
       - **Explanation**: Human-readable reasoning
       - **Recommended Action**: Next steps for security team
    
    4. **Storage**: All predictions saved for quality monitoring
    
    5. **Evaluation**: Predictions compared against labeled dataset
    
    ---
    
    ### ✨ Why This Design?
    
    **Mock Mode**: Ensures reliable demos without network dependency  
    **Lightweight**: No heavy ML frameworks (PyTorch, TensorFlow) required  
    **Modular**: Easy to swap LLM backends or storage layers  
    **Presentation-Ready**: UI serves as interactive slide deck  
    **Quality-Focused**: Built-in evaluation and monitoring from day one
    
    ---
    
    ### 🎯 How This Maps to Siemens Role
    
    | Job Requirement | PoC Feature |
    |----------------|-------------|
    | "Supporting evaluation of AI frameworks" | Framework comparison tab, abstraction layer design |
    | "Building PoC AI agents" | Functional threat detection agent with live demo |
    | "Monitoring output quality" | Metrics dashboard, confusion matrix, mistake analysis |
    | "Education in AI & Cybersecurity" | Domain-specific prompts, security pattern recognition |
    | "Working in a team" | Clean code structure, documentation, extensibility |
    
    """)
    
    # Configuration status
    st.markdown("---")
    st.subheader("⚙️ Current Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mode = "🎭 Mock Mode" if is_mock_mode() else "🌐 Real API Mode"
        st.metric("Operation Mode", mode)
    
    with col2:
        api_status = "Not Required (Mock)" if is_mock_mode() else ("✓ Configured" if settings.hf_api_key else "⚠️  Missing")
        st.metric("API Key Status", api_status)
    
    with col3:
        st.metric("Storage", "CSV")
    
    is_valid, warnings = validate_config()
    if warnings:
        for warning in warnings:
            st.warning(warning)


def render_demo_tab():
    """Render the Live Demo tab."""
    st.header("🚀 Live Demo: Threat Detection Agent")
    
    st.markdown("""
    This interactive demo lets you analyze security logs in real-time using the AI threat detection agent.
    
    **How it works:**
    1. Select example logs or paste your own
    2. Click "Analyze Logs" to run the AI agent
    3. Review predictions, explanations, and recommended actions
    4. Optionally save results for quality monitoring
    """)
    
    # Initialize agent
    agent = ThreatDetectionAgent()
    
    # Load sample logs
    sample_logs = load_sample_logs()
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Choose from examples", "Paste custom logs"],
        horizontal=True
    )
    
    if input_method == "Choose from examples":
        # Dropdown with sample logs
        selected_indices = st.multiselect(
            "Select log entries to analyze:",
            range(len(sample_logs)),
            format_func=lambda i: f"Log {i+1}: {sample_logs[i][:80]}...",
            default=[0, 4, 7]  # Default: benign, malicious brute force, SQL injection
        )
        
        logs_to_analyze = [sample_logs[i] for i in selected_indices]
    
    else:
        # Text area for custom logs
        custom_logs = st.text_area(
            "Paste log entries (one per line):",
            height=200,
            placeholder="2024-01-15 10:23:41 INFO user=alice action=login ip=192.168.1.100 status=success"
        )
        
        logs_to_analyze = [line.strip() for line in custom_logs.split('\n') if line.strip()]
    
    # Analyze button
    if st.button("🔍 Analyze Logs", type="primary", disabled=len(logs_to_analyze) == 0):
        with st.spinner("Analyzing logs with AI agent..."):
            results = agent.analyze_logs_batch(logs_to_analyze)
            
            # Store results in session state
            st.session_state['demo_results'] = results
    
    # Display results
    if 'demo_results' in st.session_state and st.session_state['demo_results']:
        results = st.session_state['demo_results']
        
        st.success(f"✓ Analyzed {len(results)} log entries")
        
        # Summary statistics
        summary = agent.get_summary_stats(results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Benign", summary['benign_count'], 
                     delta=f"{summary['benign_pct']:.1f}%", delta_color="off")
        with col2:
            st.metric("Suspicious", summary['suspicious_count'],
                     delta=f"{summary['suspicious_pct']:.1f}%", delta_color="off")
        with col3:
            st.metric("Malicious", summary['malicious_count'],
                     delta=f"{summary['malicious_pct']:.1f}%", delta_color="off")
        with col4:
            st.metric("Avg Confidence", f"{summary['avg_confidence']:.2f}")
        
        # Results table
        st.subheader("📊 Detailed Results")
        
        # Convert to DataFrame for display
        df_results = pd.DataFrame(results)
        
        # Format for display
        display_df = df_results[[
            'prediction', 'confidence', 'log', 'explanation', 'recommended_action'
        ]].copy()
        
        # Color-code predictions
        def highlight_prediction(row):
            if row['prediction'] == 'malicious':
                return ['background-color: #ffcccc'] * len(row)
            elif row['prediction'] == 'suspicious':
                return ['background-color: #fff4cc'] * len(row)
            else:
                return ['background-color: #ccffcc'] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_prediction, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        # Save to storage option
        st.markdown("---")
        if st.button("💾 Save Results to Storage"):
            store = create_store()
            ids = store.save_predictions_batch(results)
            st.success(f"✓ Saved {len(ids)} predictions to {settings.storage_path}")


def render_quality_tab():
    """Render the Quality Monitoring tab."""
    st.header("📈 Quality Monitoring")
    
    st.markdown("""
    This dashboard evaluates the AI agent's performance against a labeled dataset,
    demonstrating **continuous quality monitoring** and **human-in-the-loop workflows**.
    
    **Key Capabilities:**
    - Overall accuracy and per-class metrics
    - Confusion matrix analysis
    - False positive/negative identification
    - Interesting mistake extraction for review
    """)
    
    # Load labeled data
    try:
        store = create_store()
        labeled_df = store.load_labeled_data()
        
        st.info(f"📁 Loaded {len(labeled_df)} labeled examples from `data/labeled_logs.csv`")
        
        # Run predictions on labeled data
        if st.button("🔄 Run Evaluation", type="primary"):
            with st.spinner("Running AI agent on labeled dataset..."):
                agent = ThreatDetectionAgent()
                
                # Get predictions
                predictions = []
                for _, row in labeled_df.iterrows():
                    result = agent.analyze_log(row['log'])
                    predictions.append(result)
                
                # Add predictions to DataFrame
                labeled_df['prediction'] = [p['prediction'] for p in predictions]
                labeled_df['confidence'] = [p['confidence'] for p in predictions]
                labeled_df['explanation'] = [p['explanation'] for p in predictions]
                
                # Store in session state
                st.session_state['evaluation_df'] = labeled_df
        
        # Display evaluation results
        if 'evaluation_df' in st.session_state:
            eval_df = st.session_state['evaluation_df']
            
            # Compute metrics
            metrics = QualityMetrics()
            summary, mistakes = evaluate_predictions(eval_df)
            
            st.success("✓ Evaluation complete")
            
            # Overall accuracy
            st.subheader("🎯 Overall Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Overall Accuracy",
                    f"{summary['overall_accuracy']:.1%}",
                    help="Percentage of correctly classified logs"
                )
            
            with col2:
                st.metric(
                    "Total Samples",
                    summary['total_samples']
                )
            
            # Per-class metrics
            st.subheader("📊 Per-Class Metrics")
            
            per_class_data = []
            for cls, metrics_dict in summary['per_class_metrics'].items():
                per_class_data.append({
                    'Class': cls.capitalize(),
                    'Precision': f"{metrics_dict['precision']:.2%}",
                    'Recall': f"{metrics_dict['recall']:.2%}",
                    'F1-Score': f"{metrics_dict['f1']:.2%}",
                    'Support': metrics_dict['support']
                })
            
            st.table(pd.DataFrame(per_class_data))
            
            # Confusion matrix
            st.subheader("🔀 Confusion Matrix")
            st.markdown("*Rows = True Label, Columns = Predicted Label*")
            
            confusion = summary['confusion_matrix']
            confusion_df = pd.DataFrame(confusion).T
            
            st.dataframe(
                confusion_df.style.background_gradient(cmap='YlOrRd', axis=None),
                use_container_width=True
            )
            
            # Mistakes analysis
            st.subheader("🔍 Interesting Mistakes")
            
            tab1, tab2, tab3 = st.tabs([
                "False Negatives (Missed Threats)",
                "False Positives (False Alarms)",
                "All Errors"
            ])
            
            with tab1:
                fn_df = mistakes['false_negatives_malicious']
                if len(fn_df) > 0:
                    st.error(f"⚠️  Found {len(fn_df)} missed malicious events (critical!)")
                    st.dataframe(
                        fn_df[['log', 'true_label', 'prediction', 'explanation']],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.success("✓ No false negatives for malicious events")
            
            with tab2:
                fp_df = mistakes['false_positives_malicious']
                if len(fp_df) > 0:
                    st.warning(f"Found {len(fp_df)} false alarms")
                    st.dataframe(
                        fp_df[['log', 'true_label', 'prediction', 'explanation']],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.success("✓ No false positives for malicious classification")
            
            with tab3:
                all_errors = mistakes['all_errors']
                if len(all_errors) > 0:
                    st.info(f"Total errors: {len(all_errors)}")
                    st.dataframe(
                        all_errors[['log', 'true_label', 'prediction', 'confidence', 'explanation']],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.success("✓ Perfect classification!")
            
            # Continuous improvement notes
            st.markdown("---")
            st.subheader("🔄 Continuous Improvement Workflow")
            st.markdown("""
            **How this scales in a production Siemens environment:**
            
            1. **Data Collection**: All predictions automatically logged with timestamps
            2. **Human Review**: Security analysts review flagged events and provide labels
            3. **Quality Monitoring**: Dashboard updates daily with new metrics
            4. **Mistake Analysis**: False negatives trigger alerts for immediate review
            5. **Model Refinement**: Poor-performing cases used to improve prompts or fine-tune models
            6. **A/B Testing**: Compare different LLM backends or prompt strategies
            7. **Feedback Loop**: Analyst corrections fed back into training data
            
            This PoC demonstrates the **infrastructure** for continuous quality monitoring,
            a critical requirement for trustworthy AI in security applications.
            """)
    
    except FileNotFoundError:
        st.error("❌ Labeled dataset not found at `data/labeled_logs.csv`")
        st.info("Please ensure the data file exists to run quality evaluation.")


def render_framework_tab():
    """Render the Framework Evaluation Notes tab."""
    st.header("🔧 Framework Evaluation Notes")
    
    st.markdown("""
    This tab demonstrates the **evaluation methodology** for AI frameworks and tools
    in cybersecurity contexts, a core responsibility from the job description.
    
    While this PoC implements **one primary approach** (free LLM API + mock mode),
    the comparison below shows how to systematically evaluate alternatives.
    """)
    
    st.subheader("🔍 Evaluated Approaches")
    
    # Comparison table
    comparison_data = {
        "Approach": [
            "Free LLM API\n(Hugging Face)",
            "Self-Hosted Open LLM\n(Llama, Mistral)",
            "Commercial API\n(OpenAI, Anthropic)",
            "Rule-Based System\n(Regex patterns)",
            "Classical ML\n(Random Forest)"
        ],
        "Latency": [
            "Medium\n(200-1000ms)",
            "Low\n(50-200ms)",
            "Medium\n(100-500ms)",
            "Very Low\n(<10ms)",
            "Low\n(10-50ms)"
        ],
        "Privacy": [
            "⚠️ Data sent externally",
            "✅ Full control",
            "⚠️ Data sent externally",
            "✅ Local only",
            "✅ Local only"
        ],
        "Cost": [
            "✅ Free tier available",
            "💰 Hardware/ops costs",
            "💰 Per-token pricing",
            "✅ Minimal",
            "✅ Minimal"
        ],
        "Flexibility": [
            "High\n(prompt engineering)",
            "Very High\n(fine-tuning possible)",
            "High\n(prompt engineering)",
            "Low\n(manual rules)",
            "Medium\n(feature engineering)"
        ],
        "Maintainability": [
            "High\n(no infrastructure)",
            "Medium\n(ops overhead)",
            "High\n(no infrastructure)",
            "Low\n(brittle rules)",
            "Medium\n(retraining needed)"
        ],
        "Explanation Quality": [
            "✅ Natural language",
            "✅ Natural language",
            "✅ Natural language",
            "⚠️ Limited",
            "⚠️ Limited"
        ],
        "PoC Suitability": [
            "✅ Excellent",
            "⚠️ Requires setup",
            "✅ Good (if budget)",
            "✅ Good baseline",
            "✅ Good baseline"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("🎯 Why This PoC Uses Free LLM API + Mock Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Advantages:**
        - ✅ **Zero infrastructure**: No GPU servers, Docker, or ops overhead
        - ✅ **Free tier**: Hugging Face Inference API has generous free quota
        - ✅ **Fast iteration**: Change models by updating one URL
        - ✅ **Natural explanations**: LLMs provide human-readable reasoning
        - ✅ **Presentation-ready**: Mock mode ensures reliable demos
        """)
    
    with col2:
        st.markdown("""
        **Trade-offs:**
        - ⚠️ **Privacy**: Data sent to external API (mitigated: synthetic data in PoC)
        - ⚠️ **Latency**: Network calls slower than local inference
        - ⚠️ **Dependency**: Requires API availability (mitigated: mock mode fallback)
        - ⚠️ **Rate limits**: Free tier has request quotas
        """)
    
    st.markdown("---")
    st.subheader("🔬 Evaluation Criteria for Security AI")
    
    st.markdown("""
    When evaluating AI frameworks for cybersecurity at Siemens, consider:
    
    1. **Security & Privacy**
       - Can sensitive logs be processed locally?
       - Is data encrypted in transit/at rest?
       - Compliance with GDPR, industry standards?
    
    2. **Performance**
       - Real-time vs. batch processing requirements
       - Acceptable latency for threat detection
       - Throughput (logs per second)
    
    3. **Accuracy & Reliability**
       - False negative rate (missed threats = critical!)
       - False positive rate (alert fatigue)
       - Consistency and reproducibility
    
    4. **Operational Costs**
       - Infrastructure (GPU servers, cloud compute)
       - Licensing fees
       - Maintenance and monitoring overhead
    
    5. **Explainability**
       - Can analysts understand *why* a threat was flagged?
       - Audit trail for compliance
       - Debugging and improvement workflows
    
    6. **Integration & Scalability**
       - Fits into existing SIEM/SOC workflows?
       - Handles growing log volumes?
       - APIs for automation?
    
    This PoC prioritizes **explainability**, **fast iteration**, and **demo reliability**,
    making it ideal for initial exploration before production deployment.
    """)
    
    st.markdown("---")
    st.subheader("🚀 Next Steps in Framework Evaluation")
    
    st.markdown("""
    To extend this evaluation in a real Siemens project:
    
    1. **Benchmark Suite**: Create standardized test dataset with known threats
    2. **A/B Testing**: Compare multiple LLM backends (GPT-4, Claude, Llama) on same data
    3. **Latency Testing**: Measure p50, p95, p99 response times under load
    4. **Privacy Assessment**: Evaluate on-prem LLM options (Llama on internal servers)
    5. **Cost Analysis**: Project monthly costs at realistic log volumes (e.g., 10M logs/day)
    6. **Hybrid Approach**: Fast rule-based triage → LLM for complex cases
    
    **This PoC provides the *testing harness* to systematically compare approaches.**
    """)


def render_future_tab():
    """Render the Future Work & Siemens Fit tab."""
    st.header("🚀 Future Work & Siemens Fit")
    
    st.markdown("""
    This tab outlines how the PoC can evolve into a **production-grade multi-agent system**
    and demonstrates alignment with the Siemens working student role.
    """)
    
    st.subheader("🔮 Extension Ideas")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Multi-Agent System",
        "Human-in-the-Loop",
        "Advanced Monitoring",
        "Enterprise Integration"
    ])
    
    with tab1:
        st.markdown("""
        ### 🤖 Multi-Agent Security System
        
        Extend to multiple specialized AI agents:
        
        **1. Phishing Email Classifier**
        - Input: Email headers + body text
        - Output: Phishing risk score + explanation
        - Action: Quarantine high-risk emails, alert users
        
        **2. SIEM Alert Triage Agent**
        - Input: Raw SIEM alerts (e.g., from Splunk, QRadar)
        - Output: Priority level (P0-P4) + investigation guidance
        - Action: Route to appropriate analyst queue
        
        **3. Incident Report Drafter**
        - Input: Timeline of security events
        - Output: Draft incident report in standard format
        - Action: Accelerate analyst reporting workflow
        
        **4. Vulnerability Scanner Explainer**
        - Input: CVE IDs + scan results
        - Output: Business impact analysis + remediation priority
        - Action: Help prioritize patching
        
        **Agent Orchestration:**
        - Shared storage layer for cross-agent context
        - Event-driven triggers (e.g., malicious log → auto-draft incident report)
        - Unified quality monitoring dashboard
        """)
    
    with tab2:
        st.markdown("""
        ### 👥 Human-in-the-Loop Workflows
        
        **Analyst Feedback Interface:**
        - ✅ / ❌ buttons on each prediction
        - Free-text notes for context
        - "Request explanation" button for unclear cases
        
        **Approval Workflows:**
        - High-stakes actions (e.g., blocking IPs) require human confirmation
        - Agent provides recommendation + confidence
        - Analyst approves/rejects with reason
        
        **Active Learning:**
        - Agent flags low-confidence predictions for human labeling
        - Prioritize labeling of mistakes from quality monitoring
        - Periodically retrain or adjust prompts based on feedback
        
        **Collaborative Triage:**
        - Agent pre-sorts alerts into buckets
        - Analysts focus on "suspicious" category (most ambiguous)
        - Benign auto-archived, malicious auto-escalated
        
        **Implementation:**
        - Add `feedback` column to storage
        - Streamlit buttons for thumbs up/down
        - Weekly review meeting with metrics dashboard
        """)
    
    with tab3:
        st.markdown("""
        ### 📊 Advanced Quality Monitoring
        
        **Real-Time Dashboards:**
        - Hourly accuracy trends
        - Alert volume over time
        - False positive/negative rates by alert type
        
        **Drift Detection:**
        - Monitor input distribution (new attack patterns?)
        - Track confidence score distributions
        - Alert if performance degrades below threshold
        
        **Explainability Tools:**
        - Attention visualization (which log tokens matter most?)
        - Counterfactual explanations ("if IP was internal, would be benign")
        - Confidence calibration curves
        
        **Multi-Model Comparison:**
        - A/B test: 50% traffic to Model A, 50% to Model B
        - Champion/challenger framework
        - Statistical significance testing
        
        **Compliance Reporting:**
        - Automated weekly reports to security leadership
        - Audit logs of all AI decisions
        - Compliance with ISO 27001, NIST frameworks
        """)
    
    with tab4:
        st.markdown("""
        ### 🏢 Enterprise Integration
        
        **SIEM Integration:**
        - REST API endpoint for real-time log analysis
        - Batch processing for historical data
        - Write results back to SIEM as enriched events
        
        **Ticketing System Integration:**
        - Auto-create Jira/ServiceNow tickets for malicious events
        - Populate ticket with AI explanation + recommended actions
        - Link to relevant logs and context
        
        **Alerting & Notifications:**
        - Slack/Teams webhooks for critical threats
        - Escalation to on-call engineer for P0 incidents
        - Daily digest email with summary statistics
        
        **Authentication & Authorization:**
        - SSO integration (SAML, OAuth)
        - Role-based access control (analyst vs. admin)
        - Audit logging of all user actions
        
        **Scalability:**
        - Kubernetes deployment for auto-scaling
        - Message queue (Kafka) for high-throughput log ingestion
        - Distributed storage (PostgreSQL, ClickHouse)
        """)
    
    st.markdown("---")
    st.subheader("🎓 Job Description Alignment")
    
    st.markdown("""
    ### How This PoC Demonstrates Each Requirement
    
    | Job Requirement | Demonstrated in PoC | Future Extensions |
    |----------------|---------------------|-------------------|
    | **"Supporting evaluation of AI frameworks/tools for security"** | Framework comparison tab, abstraction layer design, mock vs. real modes | Benchmark suite, A/B testing infrastructure |
    | **"Building PoC AI agents for specific security tasks"** | End-to-end threat detection agent with live demo | Multi-agent system (phishing, SIEM triage, incident drafting) |
    | **"Monitoring and improving output quality"** | Metrics dashboard, confusion matrix, mistake analysis | Real-time drift detection, active learning, explainability tools |
    | **"Education in AI and Cybersecurity"** | Security-specific prompts, threat taxonomy, domain knowledge | Integrate MITRE ATT&CK framework, CVE databases |
    | **"Working as part of a team"** | Clean code structure, comprehensive docs, extensible design | Human-in-the-loop workflows, collaborative triage, Slack integration |
    | **"Independent and structured work"** | Complete project from architecture → implementation → testing → documentation | Milestone planning, sprint-based development |
    
    ---
    
    ### 💡 Why I'm Excited About This Role
    
    This PoC reflects my approach to AI engineering:
    
    1. **Start Small, Think Big**: Minimal viable product → scalable architecture
    2. **Quality-First**: Monitoring and evaluation built in from day one
    3. **User-Centric**: Security analysts are the users; tool must explain itself
    4. **Pragmatic**: Mock mode ensures reliability; free API reduces barriers
    5. **Team-Ready**: Clean code, docs, and extensibility for collaboration
    
    I'm eager to bring this mindset to Siemens, where I can:
    - Contribute to **real-world cybersecurity challenges**
    - Learn from experienced security professionals
    - Iterate rapidly on AI prototypes
    - Help shape responsible AI deployment in critical infrastructure
    
    **This PoC is just the beginning. I'm ready to build production systems with your team.**
    """)


def main():
    """Main application entry point."""
    
    # Sidebar
    with st.sidebar:
        st.title("🛡️ AI Threat Agent")
        st.markdown("### PoC for Siemens Interview")
        
        st.markdown("---")
        
        # Mode indicator
        if is_mock_mode():
            st.info("🎭 **Mock Mode**\nUsing deterministic pattern-based responses")
        else:
            st.success("🌐 **Real API Mode**\nCalling Hugging Face LLM")
        
        st.markdown("---")
        
        # Navigation hint
        st.markdown("""
        **Suggested Flow:**
        1. Overview & Story
        2. Live Demo
        3. Quality Monitoring
        4. Framework Evaluation
        5. Future Work
        """)
        
        st.markdown("---")
        st.caption("Built with Streamlit + Python")
        st.caption("No heavy ML frameworks required")
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 Overview & Story",
        "🚀 Live Demo",
        "📈 Quality Monitoring",
        "🔧 Framework Evaluation",
        "🔮 Future Work & Siemens Fit"
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
