# data_analyst_agent

Multi-agent **AI Data Analyst** built with LangGraph. One CSV goes in; a validated
analysis report (HTML + PDF) and an interactive, report-grounded Q&A come out.

## Architecture & Frontends

Pipeline: **Profiler → Analysis → Insight & Report**, joined by a single shared `AgentState` contract.

| Component | Role | Code / Directory | Details |
|-----------|------|------------------|---------|
| **Member 1 — Profiler** | CSV → pandas + sweetviz HTML report + structured `profile` | `agents/profiler/`, `tools/` | Ingests CSV & runs EDA |
| **Member 2 — Analysis** | planner → executor (sandboxed code) → reflector → `analysis_results` | `agents/analysis/` | Executes statistical tasks |
| **Member 3 — Insight** | validates, writes insights/recommendations, compiles HTML+PDF report | `agents/insight/` | Generates final reports & Q&A |
| **FastAPI Backend** | REST API orchestrating the multi-agent graph & chat endpoints | `api/main.py` | Runs on port 8000 |
| **React Web Frontend** | Modern TanStack / Vite dashboard for pipeline launch & analysis | `frontend/` | Built with React & Tailwind |
| **Streamlit Dashboard** | Alternative Python UI dashboard | `app.py` / `ui/` | Classic Streamlit UI |

## Quick start

### 1. Run FastAPI Backend
```bash
python -m uvicorn api.main:app --reload --port 8000
```

### 2. Run React Frontend
```bash
cd frontend
npm install
npm run dev
```

### 3. Run Streamlit App (Alternative UI)
```bash
streamlit run app.py
```

### 4. Run Test Suite
```bash
pytest
```

## Shared state contract
Single source of truth: `state.py` (or `state/graph_state.py`) re-exported so all agents import `AgentState` from one place.

## Setup / env
Set the relevant LLM keys in `.env`: `OPENAI_API_KEY`, `GROQ_API_KEY`, or `GEMINI_API_KEY`.