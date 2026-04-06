# ContentBlitz — Tech Stack

---

## Core Framework

| Component | Technology | Purpose |
|---|---|---|
| Multi-Agent Orchestration | **LangGraph** | Defines agent graph, state management, conditional routing between agents |
| Language Model | **OpenAI GPT-4** | Powers all text generation — research synthesis, blog writing, LinkedIn posts, query classification |
| Image Generation | **DALL-E 3** (via OpenAI API) | Generates visual content from optimized prompts |
| Web Research | **SERP API** + GPT | Web search for deep research, fact gathering, source discovery |
| Web Interface | **Streamlit** | Chat-based UI, content dashboard, export controls |

---

## Alternative / Fallback Providers

| Component | Alternatives |
|---|---|
| Language Model | Claude Sonnet, Google Gemini |
| Research Engine | Perplexity Sonar, Tavily AI, You.com Search |
| Image Generation | Stability AI, Midjourney API, Google Imagen |
| Content Optimization | Jasper AI, Copy.ai, Writesonic |
| Web Interface | Gradio, React, Flask + HTML/CSS |
| State Management | Redis, MongoDB, PostgreSQL |

---

## Python Libraries & Dependencies

| Library | Version (approx) | Purpose |
|---|---|---|
| `langgraph` | latest | Multi-agent workflow graph, state management |
| `langchain-core` | latest | Base abstractions for LLM chains and prompts |
| `langchain-openai` | latest | OpenAI LLM and embedding integrations |
| `openai` | >=1.0 | Direct OpenAI API client (GPT-4, DALL-E 3) |
| `google-search-results` | latest | SERP API Python client |
| `streamlit` | >=1.30 | Web UI framework |
| `pydantic` | >=2.0 | Data validation, state schema definitions |
| `python-dotenv` | latest | Load `.env` environment variables |
| `pyyaml` | latest | Parse YAML config files |
| `httpx` | latest | Async HTTP client for API calls |
| `tenacity` | latest | Retry logic with exponential backoff |
| `pytest` | latest | Testing framework |
| `pytest-cov` | latest | Code coverage reporting |
| `pytest-asyncio` | latest | Async test support |

---

## Infrastructure & DevOps

| Tool | Purpose |
|---|---|
| **Docker** + `docker-compose` | Containerized deployment |
| **Python 3.11+** | Runtime |
| **pip** / **venv** | Dependency management |
| **YAML configs** | Environment-specific settings (dev/prod) |
| **Structured logging** (`logging` stdlib) | Observability across all modules |

---

## Optional / Future Integrations

| Tool | Purpose |
|---|---|
| **LangSmith** | LLM call monitoring, debugging, tracing |
| **Weights & Biases** | Experiment tracking |
| **Helicone** | LLM cost observability |
| **LangGraph Cloud** | Serverless agent deployment |
| **Modal** | GPU-accelerated cloud functions |

---

## API Keys Required

| Key | Service | Required? |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI (GPT-4 + DALL-E 3) | **Yes** |
| `SERP_API_KEY` | SerpApi (web search) | **Yes** |
| `PERPLEXITY_API_KEY` | Perplexity Sonar (fallback research) | Optional |
| `LANGSMITH_API_KEY` | LangSmith (monitoring) | Optional |
