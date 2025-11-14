# AI Threat Detection Agent - PoC

> **A lightweight, demo-ready proof-of-concept for LLM-based cybersecurity threat detection**
> Built for the Siemens AI/Cybersecurity Working Student Interview

## 🎯 Project Overview

This PoC demonstrates an AI-powered security agent that analyzes log files to detect threats, explains its reasoning, and monitors its own quality — showcasing the key competencies from the job description:

- ✅ **Evaluating AI frameworks/tools** (comparison of LLM APIs, self-hosted models, baselines)
- ✅ **Building PoC AI agents** for specific security tasks (threat classification + explanation)
- ✅ **Monitoring output quality** (metrics dashboard, human-in-the-loop evaluation)

**Key Features:**
- 🚀 Interactive web UI that doubles as a live presentation
- 🔌 Free LLM API integration (Hugging Face Inference API) with mock mode for offline demo
- 📊 Quality monitoring dashboard with accuracy metrics and mistake analysis
- 💾 Lightweight storage (CSV-based, easily switchable to SQLite)
- 🧪 Minimal dependencies, runs in seconds

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- (Optional) Hugging Face API key for live LLM mode

### Installation & Run

```bash
# 1. Set up the environment
make setup

# 2. (Optional) Configure API key for live mode
cp .env.example .env
# Edit .env and add your HF_API_KEY (or leave blank for mock mode)

# 3. Run the application
make run
```

The UI will open at `http://localhost:8501`

### Alternative: Manual Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/ui_app.py
```

---

## 🎤 How to Present This in an Interview

**Suggested Flow (15-20 minutes):**

1. **Start with "Overview & Story" tab** (3 min)
   - Walk through the project purpose and architecture
   - Explain how it maps to the job description
   - Show the simple pipeline diagram

2. **Live Demo tab** (5 min)
   - Select example logs from the dropdown
   - Click "Analyze Logs" to show real-time threat detection
   - Highlight the structured output: prediction + explanation + recommended action
   - Emphasize mock mode enables offline demo (reliability!)

3. **Quality Monitoring tab** (4 min)
   - Show metrics dashboard (accuracy, confusion matrix)
   - Point out false negatives/positives
   - Explain how this enables continuous improvement
   - Discuss human-in-the-loop workflows

4. **Framework Evaluation Notes tab** (3 min)
   - Walk through the comparison table
   - Justify architectural decisions (why free API + mock mode)
   - Show you understand tradeoffs (latency, privacy, cost, control)

5. **Future Work & Siemens Fit tab** (3 min)
   - Outline 3-4 extension ideas (phishing detection, SIEM triage, etc.)
   - Connect each job requirement to project features
   - Show enthusiasm for team-based, iterative AI development

6. **Q&A / Code Walkthrough** (remaining time)
   - Offer to dive into `llm_client.py` abstraction
   - Discuss testing strategy (`make test`)
   - Explain how to swap storage backends or add new frameworks

**Key Talking Points:**
- "This PoC intentionally stays minimal to focus on the evaluation methodology"
- "Mock mode makes this presentation-reliable and demonstrates engineering discipline"
- "The storage layer captures all predictions for continuous quality monitoring"
- "I designed the UI as a storytelling tool, not just a technical demo"

---

## 🏗️ Architecture

```
┌─────────────┐
│   Web UI    │  Streamlit-based interactive presentation
│ (ui_app.py) │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Threat Agent    │  Orchestrates detection pipeline
│(threat_agent.py) │
└────┬─────────────┘
     │
     ├──────────────────┐
     ▼                  ▼
┌─────────────┐   ┌─────────────┐
│ LLM Client  │   │  Storage    │
│ • Real Mode │   │ (CSV/SQLite)│
│ • Mock Mode │   │             │
└─────────────┘   └─────────────┘
     │
     ▼
┌──────────────────────┐
│  HuggingFace API     │  Free-tier LLM
│  (or mock responses) │
└──────────────────────┘
```

---

## 🧪 Testing

```bash
# Run unit tests
make test

# Or manually
pytest tests/ -v
```

Tests cover:
- Metrics calculation (accuracy, confusion matrix)
- Threat agent logic in mock mode
- Deterministic mock responses

---

## 📂 Project Structure

```
src/
├── config.py          # Environment configuration (API keys, mock mode flag)
├── llm_client.py      # LLM API abstraction (real + mock implementations)
├── threat_agent.py    # Core threat detection logic
├── storage.py         # CSV/SQLite persistence layer
├── metrics.py         # Quality monitoring metrics
└── ui_app.py          # Streamlit UI with all presentation tabs

data/
├── sample_logs.txt    # Example logs for live demo
└── labeled_logs.csv   # Ground truth dataset for quality monitoring

tests/
├── test_metrics.py    # Unit tests for evaluation metrics
└── test_threat_agent.py  # Tests for agent logic
```

---

## 🔧 Configuration

Edit `.env` or set environment variables:

```bash
# LLM API Configuration
HF_API_KEY=your_huggingface_api_key_here
HF_API_URL=https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2

# Mode Selection
USE_MOCK_MODEL=true  # Set to 'false' for real API calls

# Storage
STORAGE_PATH=data/predictions.csv
```

**Mock Mode (default):**
Returns deterministic responses based on pattern matching. Perfect for demos without network dependency.

**Real Mode:**
Calls Hugging Face Inference API with the configured model. Requires valid `HF_API_KEY`.

---

## 🎓 Learning Outcomes & Job Alignment

This PoC directly demonstrates skills from the job description:

| Job Requirement | How This PoC Addresses It |
|----------------|---------------------------|
| "Supporting evaluation of AI frameworks/tools" | Framework comparison tab, abstraction layer design |
| "Building PoC AI agents for security tasks" | End-to-end threat detection agent with real output |
| "Monitoring output quality" | Metrics dashboard, labeled dataset evaluation |
| "Education in AI & Cybersecurity" | Integrated domain knowledge in prompt engineering |
| "Team collaboration" | Clear code structure, documentation, extensibility |

---

## 🚧 Future Extensions

- **Multi-agent system**: Add phishing email classifier, SIEM alert triage
- **Human-in-the-loop**: Feedback UI for analysts to correct predictions
- **Advanced metrics**: ROC curves, per-threat-type precision/recall
- **Real-time streaming**: Integrate with log ingestion pipelines
- **Model comparison**: A/B test different LLMs or fine-tuned models

---

## 📜 License

This is a portfolio/interview project. Feel free to use as reference material.

---

## 👤 Author

Created as a technical demonstration for the Siemens AI/Cybersecurity Working Student position.
