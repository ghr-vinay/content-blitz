# ContentBlitz - AI Content Marketing Assistant

---

## Core Focus Areas

These are the primary capabilities this project demonstrates:

1. **Multi-Agent Orchestration with LangGraph** — Build a graph of specialized agents with conditional routing, shared state, and coordinated workflows
2. **LangChain Tool Usage** — Agents leverage LangChain tools (web search, API calls, content generation) to accomplish their tasks
3. **Image Generation** — AI-powered visual content creation via DALL-E 3 with prompt optimization
4. **Basic UI for User Interaction** — Streamlit chat interface for users to converse with the agent system and view outputs
5. **Logging, Monitoring & Observability via LangSmith** — Trace every LLM call, agent step, and tool invocation for debugging and performance analysis
6. **LLM Evaluation (LLM-as-Judge)** — Automated content quality scoring using an LLM to evaluate generated outputs against defined criteria
7. **Deployment** _(optional)_ — Docker containerization and cloud deployment

> All tasks below are ordered to prioritize these core areas first. Nice-to-have features are grouped at the end.

---

## SOLID Design Principles

The codebase will follow SOLID principles throughout:

| Principle | How It's Applied |
|---|---|
| **S — Single Responsibility** | Each agent has exactly one job (research, blog writing, image generation, etc.). Integration clients handle only API communication. Config, routing, and workflow are separate modules. |
| **O — Open/Closed** | A base `BaseAgent` ABC defines the agent contract. New agents can be added by subclassing without modifying existing code. Tools are registered via a plugin-style list — extend by adding, not editing. |
| **L — Liskov Substitution** | All agents implement the same `BaseAgent` interface and can be swapped interchangeably in the LangGraph graph. Any `BaseTool` implementation can replace another wherever a tool is expected. |
| **I — Interface Segregation** | Agents depend only on the interfaces they need — a blog writer doesn't know about image generation internals. Tool interfaces are small and focused (e.g., `SearchTool`, `ImageTool` — not one giant `AITool`). |
| **D — Dependency Inversion** | Agents depend on abstractions (base classes / protocols), not concrete implementations. LLM clients, search tools, and image generators are injected via config, not hardcoded. Swapping OpenAI for another provider requires zero agent code changes. |

---

## Task Breakdown

---

### Phase 1: Project Setup & Infrastructure

- [x] **1.1** Initialize Python project structure (`src/agents/`, `src/core/`, `src/integrations/`, `src/web_app/`, `src/utils/`, `src/workflow/`, `tests/`, `config/`, `docs/`)
- [x] **1.2** Create `requirements.txt` — langgraph, langchain, langchain-openai, openai, streamlit, google-search-results, langsmith, python-dotenv, pyyaml, pydantic, tenacity, httpx, pytest
- [x] **1.3** Create `.env.example` with all required API keys (`OPENAI_API_KEY`, `SERP_API_KEY`, `LANGSMITH_API_KEY`, etc.)
- [x] **1.4** Set up `config/development.yaml` and `config/production.yaml` for environment-specific settings
- [x] **1.5** Set up `config/services.yaml` for AI service provider configuration
- [x] **1.6** Set up logging framework with structured logging across all modules
- [x] **1.7** Create `__init__.py` files for all packages
- [x] **1.8** Define `src/agents/base_agent.py` — abstract base class (`BaseAgent`) with `run(state) -> state` contract that all agents must implement (**Open/Closed, Liskov Substitution**)
- [x] **1.9** Define `src/integrations/base_tool.py` — abstract base class / protocol for all tool integrations (search, image, LLM) so agents depend on abstractions, not concrete clients (**Dependency Inversion, Interface Segregation**)
- [x] **1.10** Define `src/core/models.py` — shared Pydantic models for agent inputs/outputs (e.g., `ResearchResult`, `BlogPost`, `LinkedInPost`, `ImageResult`) to enforce clear data contracts (**Single Responsibility**)

---

### Phase 2: Core Configuration & LangSmith Setup

- [x] **2.1** Implement `src/core/config.py` — centralized config loader (reads env vars, YAML configs, validates required keys)
- [x] **2.2** Configure LangSmith tracing — enable `LANGCHAIN_TRACING_V2`, set project name, wire tracing into all LLM calls and agent steps
- [x] **2.3** Verify LangSmith dashboard shows traces for a simple LLM call (smoke test)

---

### Phase 3: Integration Clients (LangChain Tools)

- [ ] **3.1** Implement `src/integrations/openai_client.py` — OpenAI GPT-4 wrapper implementing `BaseLLMClient` interface using LangChain ChatOpenAI (**Dependency Inversion**)
- [ ] **3.2** Implement `src/integrations/serp_client.py` — SERP API wrapper implementing `BaseSearchTool` interface as a LangChain Tool (**Interface Segregation**)
- [ ] **3.3** Implement `src/integrations/image_clients.py` — DALL-E 3 implementation of `BaseImageTool` interface as a LangChain Tool (**Liskov Substitution** — can be swapped with any other image provider)
- [ ] **3.4** Implement retry/error handling across all integration clients (exponential backoff via tenacity)
- [ ] **3.5** _(Optional)_ Implement `src/integrations/perplexity_client.py` — implements same `BaseSearchTool` interface as SERP client, demonstrating provider swappability (**Open/Closed**)

---

### Phase 4: LangGraph State & Workflow Engine

- [ ] **4.1** Implement `src/workflow/state_management.py` — define LangGraph `TypedDict` state schema (user query, intent, research context, generated content, conversation history, image URLs)
- [ ] **4.2** Implement `src/workflow/langgraph_workflow.py` — build the LangGraph `StateGraph` with nodes for each agent and conditional edges for routing
- [ ] **4.3** Implement conversation memory — persist multi-turn context across interactions using LangGraph state
- [ ] **4.4** Implement error handling within the graph — graceful degradation, fallback nodes, partial result recovery
- [ ] **4.5** Verify full graph traces appear in LangSmith with node-level visibility

---

### Phase 5: Agent Implementation

All agents extend `BaseAgent` and implement the `run(state) -> state` contract. Each agent has a **single responsibility** and receives its dependencies (LLM client, tools) via constructor injection (**Dependency Inversion**).

#### 5A: Query Handler Agent (Router)
- [ ] **5.1** Implement `src/agents/query_handler.py` — extends `BaseAgent`; single responsibility is intent classification (research, blog, linkedin, image, strategy)
- [ ] **5.2** Build routing logic — map classified intent to the correct downstream agent node(s) in the graph
- [ ] **5.3** Handle ambiguous queries — ask clarifying questions or select a reasonable default
- [ ] **5.4** Support multi-intent queries — detect when user wants multiple outputs (e.g., "research X and write a blog about it")

#### 5B: Deep Research Agent
- [ ] **5.5** Implement `src/agents/research_agent.py` — extends `BaseAgent`; depends on injected `BaseSearchTool`, not a concrete SERP client (**DIP**)
- [ ] **5.6** Add research synthesis — aggregate multiple search results into a coherent research summary
- [ ] **5.7** Add source attribution — track and include URLs/references in research output

#### 5C: SEO Blog Writer Agent
- [ ] **5.8** Implement `src/agents/blog_writer.py` — extends `BaseAgent`; single responsibility is SEO blog generation, outputs `BlogPost` Pydantic model
- [ ] **5.9** Integrate keyword research — extract primary/secondary keywords and weave them into content
- [ ] **5.10** Generate meta descriptions, title tags, header hierarchy (H1/H2/H3)
- [ ] **5.11** Accept research output as input context for research-first workflows

#### 5D: LinkedIn Post Writer Agent
- [ ] **5.12** Implement `src/agents/linkedin_writer.py` — extends `BaseAgent`; single responsibility is LinkedIn post generation, outputs `LinkedInPost` model
- [ ] **5.13** Add hashtag strategy — generate relevant hashtags based on content and trends
- [ ] **5.14** Optimize for LinkedIn formatting (character limits, line breaks, hooks)

#### 5E: Image Generation Agent
- [ ] **5.15** Implement `src/agents/image_generator.py` — extends `BaseAgent`; depends on injected `BaseImageTool`, not DALL-E directly (**DIP**)
- [ ] **5.16** Add prompt optimization — enhance user's vague image request into a detailed DALL-E prompt
- [ ] **5.17** Support style/size parameters — aspect ratio, style (photorealistic, illustration, etc.)
- [ ] **5.18** Handle image generation errors gracefully (content policy violations, rate limits)

#### 5F: Content Strategist Agent
- [ ] **5.19** Implement `src/agents/content_strategist.py` — extends `BaseAgent`; single responsibility is formatting/organizing research into structured content

---

### Phase 6: Router & Orchestration

- [ ] **6.1** Implement `src/core/router.py` — routing logic maps intent to agent nodes; references agents by their `BaseAgent` interface, not concrete classes (**DIP**)
- [ ] **6.2** Implement `src/core/workflow.py` — high-level workflow orchestration (research → blog, research → linkedin, etc.); adding new workflows doesn't modify existing ones (**OCP**)
- [ ] **6.3** Support chained workflows — e.g., research first, then generate blog + linkedin + image from the same research
- [ ] **6.4** Verify all multi-agent flows are fully traced end-to-end in LangSmith

---

### Phase 7: Streamlit Web Interface

- [ ] **7.1** Implement `src/web_app/streamlit_app.py` — main chat-based UI with conversation history
- [ ] **7.2** Build chat input component — natural language input with send button
- [ ] **7.3** Build content display components — render blog posts, LinkedIn posts, images, research summaries with proper formatting
- [ ] **7.4** Add sidebar controls — select content type, view agent routing info
- [ ] **7.5** Handle loading states and streaming responses for long-running agent tasks
- [ ] **7.6** Add export/download buttons — download generated content as markdown or copy to clipboard

---

### Phase 8: LLM Evaluation (LLM-as-Judge)

- [ ] **8.1** Implement `src/utils/llm_eval.py` — LLM-as-Judge evaluator that scores generated content on relevance, coherence, accuracy, and completeness
- [ ] **8.2** Define evaluation criteria/rubrics for each content type (blog, LinkedIn post, research summary)
- [ ] **8.3** Integrate evaluation into the workflow — auto-score outputs after generation and surface scores in the UI
- [ ] **8.4** Log evaluation scores to LangSmith for tracking quality trends over time
- [ ] **8.5** _(Optional)_ Implement pairwise comparison evaluator — compare two versions of content and pick the better one

---

### Phase 9: Observability & Monitoring Polish

- [ ] **9.1** Add custom LangSmith metadata to traces — tag by agent type, content type, user session
- [ ] **9.2** Track latency per agent node — identify slow steps in the graph
- [ ] **9.3** Track token usage and estimated cost per request via LangSmith
- [ ] **9.4** Set up LangSmith dataset + evaluator runs for regression testing content quality

---

### Phase 10: Testing

- [ ] **10.1** Write unit tests for each agent (`tests/unit/`) — inject mock tools/LLM via constructor (easy to test thanks to **DIP**), test prompt construction, validate output Pydantic models
- [ ] **10.2** Write unit tests for integration clients — test each implements its base interface contract, test retry logic, error handling
- [ ] **10.3** Write unit tests for LLM eval module — test scoring logic with canned inputs
- [ ] **10.4** Write integration tests (`tests/integration/`) — test full agent → tool → LLM pipeline
- [ ] **10.5** Write E2E tests (`tests/e2e/`) — test full workflow from user query → routed agent → generated content → eval score

---

### Phase 11: Documentation

- [ ] **11.1** Write comprehensive `README.md` — overview, architecture, setup instructions, usage examples, troubleshooting
- [ ] **11.2** Write `docs/architecture.md` — system architecture decisions, LangGraph design, agent design rationale
- [ ] **11.3** Write `docs/api_documentation.md` — document each agent's input/output schema, tool configurations
- [ ] **11.4** Add inline docstrings to all public functions and classes

---

### Phase 12 (OPTIONAL): Advanced Features & Deployment

- [ ] **12.1** Add multi-turn conversation refinement — "make it more formal", "add more statistics", etc.
- [ ] **12.2** Add content series generation — create related content pieces for a campaign from one topic
- [ ] **12.3** Brand voice consistency — apply tone/style guidelines across all output
- [ ] **12.4** Content quality enhancement pipeline — post-generation optimization pass
- [ ] **12.5** SEO scoring — rate blog output on keyword density, readability, structure
- [ ] **12.6** Platform-specific formatting validation (LinkedIn char limits, blog structure)
- [ ] **12.7** Response caching layer to reduce redundant API calls and control costs
- [ ] **12.8** `docker-compose.yml` + Dockerfile for containerized deployment
- [ ] **12.9** Write `docs/deployment_guide.md` — Docker setup, env configuration, cloud deployment options
- [ ] **12.10** CMS integration — publish directly to WordPress/Ghost/Medium
- [ ] **12.11** Social media scheduling — integrate with Buffer/Hootsuite
- [ ] **12.12** Fact-checking heuristics — cross-reference claims across multiple sources

---

## Dependency Graph (Suggested Order)

```
Phase 1  (Project Setup)
  → Phase 2  (Config & LangSmith Setup)
  → Phase 3  (Integration Clients / LangChain Tools)
  → Phase 4  (LangGraph State & Workflow)
  → Phase 5  (All 6 Agents)
  → Phase 6  (Router & Orchestration)
  → Phase 7  (Streamlit UI)
  → Phase 8  (LLM Evaluation)
  → Phase 9  (Observability Polish)
  → Phase 10 (Testing)
  → Phase 11 (Documentation)
  → Phase 12 (Optional: Advanced Features & Deployment)
```

**Total: ~55 core tasks + ~12 optional tasks across 12 phases**
