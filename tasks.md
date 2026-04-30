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

- [x] **3.1** Implement `src/integrations/openai_client.py` — OpenAI GPT-4 wrapper implementing `BaseLLMClient` interface using LangChain ChatOpenAI (**Dependency Inversion**)
- [x] **3.2** Implement `src/integrations/serp_client.py` — SERP API wrapper implementing `BaseSearchTool` interface as a LangChain Tool (**Interface Segregation**)
- [x] **3.3** Implement `src/integrations/image_clients.py` — DALL-E 3 implementation of `BaseImageTool` interface as a LangChain Tool (**Liskov Substitution** — can be swapped with any other image provider)
- [x] **3.4** Implement retry/error handling across all integration clients (exponential backoff via tenacity)
- [x] **3.5** _(Optional)_ Implement `src/integrations/perplexity_client.py` — implements same `BaseSearchTool` interface as SERP client, demonstrating provider swappability (**Open/Closed**)

---

### Phase 4: LangGraph State & Workflow Engine

- [x] **4.1** Implement `src/workflow/state_management.py` — define LangGraph `TypedDict` state schema (user query, intent, research context, generated content, conversation history, image URLs); includes `remaining_nodes: list[str]` — a per-request agent execution queue populated by `QueryHandlerAgent` and consumed by `_advance_queue` to drive generalised routing
- [x] **4.2** Implement `src/workflow/langgraph_workflow.py` — build the LangGraph `StateGraph` with nodes for each agent and conditional edges for routing
- [x] **4.2.1** *(Improvement)* Generalise routing — replaced all per-intent conditional edge functions (`_route_after_research`, `_route_after_blog`, etc.) with a single `_route_next(state)` that reads `state["remaining_nodes"][0]`; added `_advance_queue(agent_fn)` wrapper that pops the head of the queue after each agent runs, keeping agents fully unaware of routing; `build_graph` now registers a single conditional edge for every node using one shared destination map — adding a new intent chain requires only a new `_INTENT_TO_NODE` entry, zero routing code changes needed (**OCP**)
- [x] **4.3** Implement conversation memory — persist multi-turn context across interactions using LangGraph state; messages serialized with correct role (`HumanMessage` / `AIMessage`) for accurate LLM context; prior history reconstructed in correct chronological order before the current query
- [x] **4.4** Implement error handling within the graph — graceful degradation, fallback nodes, partial result recovery
- [x] **4.5** Verify full graph traces appear in LangSmith with node-level visibility

---

### Phase 5: Agent Implementation

All agents extend `BaseAgent` and implement the `run(state) -> state` contract. Each agent has a **single responsibility** and receives its dependencies (LLM client, tools) via constructor injection (**Dependency Inversion**).

#### 5A: Query Handler Agent (Router)
- [x] **5.1** Implement `src/agents/query_handler.py` — extends `BaseAgent`; single responsibility is intent classification (research, blog, linkedin, image, strategy)
- [x] **5.2** Build routing logic — `QueryHandlerAgent` writes `_INTENT_TO_NODE[intent]` into `state["remaining_nodes"]`; the graph's generalised `_route_next` function follows this queue node-by-node; `_INTENT_TO_NODE` is the single source of truth for all agent chains
- [x] **5.3** Handle ambiguous queries — ask clarifying questions or select a reasonable default
- [x] **5.4** Support multi-output intents — `blog_with_image` (research → blog → image) and `linkedin_with_image` (research → linkedin → image) replace the former `multi` intent; `blog` and `linkedin` always include research automatically; `_VALID_INTENTS` and system prompt updated with examples for all new intents
- [x] **5.5** Inject sliding window conversation history into query_handler prompt — last `_HISTORY_WINDOW` (6) messages formatted as `User`/`Assistant` lines; enables follow-up resolution (e.g. "now write a blog about that") without exceeding token limits

#### 5B: Deep Research Agent
- [x] **5.6** Implement `src/agents/research_agent.py` — extends `BaseAgent`; depends on injected `BaseSearchTool`, not a concrete SERP client (**DIP**)
- [x] **5.7** Add research synthesis — aggregate multiple search results into a coherent research summary
- [x] **5.8** Add source attribution — track and include URLs/references in research output; Also persist raw source snippets for source quality evaluation at later stages.

#### 5C: SEO Blog Writer Agent
- [x] **5.9** Implement `src/agents/blog_writer.py` — extends `BaseAgent`; single responsibility is SEO blog generation, outputs `BlogPost` Pydantic model
- [x] **5.10** Integrate keyword research — extract primary/secondary keywords and weave them into content
- [x] **5.11** Generate meta descriptions, title tags, header hierarchy (H1/H2/H3)
- [x] **5.12** Accept research output as input context for research-first workflows

#### 5D: LinkedIn Post Writer Agent
- [x] **5.12** Implement `src/agents/linkedin_writer.py` — extends `BaseAgent`; single responsibility is LinkedIn post generation, outputs `LinkedInPost` model
- [x] **5.13** Add hashtag strategy — generate relevant hashtags based on content and trends
- [x] **5.14** Optimize for LinkedIn formatting (character limits, line breaks, hooks)

#### 5E: Image Generation Agent
- [x] **5.15** Implement `src/agents/image_generator.py` — extends `BaseAgent`; depends on injected `BaseImageTool`, not DALL-E directly (**DIP**)
- [x] **5.16** Add prompt optimization — enhance user's vague image request into a detailed DALL-E prompt
- [x] **5.17** Support style/size parameters — aspect ratio, style (photorealistic, illustration, etc.)
- [x] **5.18** Handle image generation errors gracefully (content policy violations, rate limits)

#### 5F: Content Strategist Agent
- [x] **5.19** Implement `src/agents/content_strategist.py` — extends `BaseAgent`; single responsibility is formatting/organizing research into structured content

---

### Phase 6: Router & Orchestration

- [x] **6.1** Implement `src/core/router.py` — routing logic maps intent to agent nodes; references agents by their `BaseAgent` interface, not concrete classes (**DIP**)
- [x] **6.2** Implement `src/core/workflow.py` — high-level workflow orchestration (research → blog, research → linkedin, etc.); adding new workflows doesn't modify existing ones (**OCP**)
- [x] **6.3** Support chained workflows — e.g., research first, then generate blog + linkedin + image from the same research
- [x] **6.4** Verify all multi-agent flows are fully traced end-to-end in LangSmith
- [x] **6.5** Implement `src/cli_app.py` — interactive terminal interface for back-and-forth conversations with the agent system; supports multi-turn history, `/clear`, `/quit`, and formatted output per content type
- [x] **6.6** Implement `src/visualise_app.py` — renders the compiled LangGraph as a PNG via the Mermaid.js API, saves to `graph.png`, and auto-opens it in default image-viewer; falls back to ASCII terminal output if the render fails
---

### Phase 7: Iterative Refinement & Context-Aware Query Enrichment

- [x] **7.1** Add `clarified_user_query: Optional[str]` to `GraphState` and `AgentState` — enriched cumulative query produced by `QueryHandlerAgent`; downstream agents use this instead of `user_query` (original is preserved for logging)
- [x] **7.2** Add `is_refinement: bool` to `GraphState` and `AgentState` — `True` when `QueryHandlerAgent` detects the user is extending or updating prior output (keywords: also, add, include, update, expand, revise, etc.)
- [x] **7.2.1** *(Bug fix)* Declare `linkedin_post_type`, `image_style`, and `image_size` as proper fields in `GraphState` — previously injected with `# type: ignore[typeddict-item]` as undeclared keys, which caused LangGraph to silently drop them after the first node executed; also added to `AgentState` and wired through `_graph_state_to_agent_state`
- [x] **7.3** Update `QueryHandlerAgent` to output `clarified_user_query` (full cumulative merged topic) and `is_refinement`; add fast-path heuristic to catch obvious refinement signals the LLM might miss
- [x] **7.4** Update `ResearchAgent` — when `is_refinement=True`, merge new search results with existing `ResearchResult` using a dedicated refinement synthesis prompt instead of regenerating from scratch
- [x] **7.5** Update `BlogWriterAgent` — when `is_refinement=True`, pass existing `BlogPost` content into a refinement prompt to patch/extend the post rather than rewrite it
- [x] **7.6** Update `LinkedInWriterAgent` — when `is_refinement=True`, pass existing `LinkedInPost` content into a refinement prompt to update the post
- [x] **7.7** Update `ImageGeneratorAgent` — when `is_refinement=True`, pass existing `ImageResult.prompt_used` into a refinement prompt optimizer to produce an updated DALL-E prompt
- [x] **7.8** *(Improvement)* Fix research synthesis ignoring user-specific requirements (e.g. "include examples") — `ResearchAgent` now runs a lightweight LLM call (`_extract_search_queries`) to convert the full `clarified_user_query` into 2-3 targeted keyword-style SERP queries covering distinct angles of the request; results from all queries are merged and deduplicated before synthesis; `_SYNTHESIS_PROMPT` and `_REFINEMENT_SYNTHESIS_PROMPT` updated to receive the full `user_request` and include an explicit rule that the summary MUST address all aspects of the request, not just the general topic
---

### Phase 8: Streamlit Web Interface

- [x] **8.1** Implement `src/web_app/streamlit_app.py` — main chat-based UI with conversation history
- [x] **8.2** Build chat input component — natural language input with send button
- [x] **8.3** Build content display components — render blog posts, LinkedIn posts, images, research summaries with proper formatting
- [x] **8.4** Add sidebar controls — image style, LinkedIn post type, image size selectors; clear conversation button
- [x] **8.5** Handle loading states — spinner during agent workflow execution
- [x] **8.6** Add export/download buttons — download generated content as markdown or copy to clipboard
- [x] **8.7** *(Improvement)* Add zero-state suggestion cards — when conversation is empty, show 3 randomly sampled prompt suggestions centered vertically and horizontally on the page; clicking a card submits the prompt via `st.session_state["_pending_prompt"]` + `st.rerun()`; cards disappear once the first message is sent
- [x] **8.7.1** *(Bug fix)* `StreamlitDuplicateElementKey` crash when the same topic appears in multiple chat history messages — all `_download_button` and `st.button` widget keys were derived solely from content hashes (e.g. `dl_research_{hash(r.topic)}`), which collide when history is re-rendered; added `msg_idx: int = 0` parameter to `_render_result`, `_render_research`, `_render_blog`, `_render_linkedin`, and `_render_strategy`; all widget keys now include `msg_idx` as a prefix; `_render_chat_history` passes the enumerate index and the in-flight render passes `len(st.session_state.messages)` to guarantee global uniqueness

---

### Phase 9: FallbackAgent & Off-Topic Handling

- [x] **9.1** Create `src/agents/fallback_agent.py` — `FallbackAgent` that generates a polite LLM response telling the user their request is out of scope or unclear, and redirects them to ContentBlitz's supported capabilities
- [x] **9.2** Add `off_topic` intent to `QueryHandlerAgent` — update `_VALID_INTENTS`, `_SYSTEM_PROMPT` (Rule 3 + examples) so the classifier can detect and route off-topic or ambiguous requests
- [x] **9.3** Add `fallback_message: Optional[str]` to `GraphState` and `AgentState`; initialise to `None` in `initial_state`; map through `_graph_state_to_agent_state`
- [x] **9.4** Wire `fallback_agent` node into `langgraph_workflow.py` — register node, add `"off_topic"` branch in `_INTENT_TO_NODE`, add terminal edge to `END`; inject via `router.py`
- [x] **9.5** Render `fallback_message` in CLI (`cli_app.py`) and Streamlit (`streamlit_app.py`) — early-return path before normal content renderers
- [x] **9.5.1** *(Bug fix)* `FallbackAgent` not firing for conversational filler — `QueryHandlerAgent` Rule 2 previously defaulted *all* ambiguous input (including `"hm"`, `"ok"`, `"ah i see"`) to `"research"` using conversation history to re-derive a topic; added an explicit rule and examples for short acknowledgements / reactions → `"off_topic"` *before* the ambiguity-default rule so these messages never reach the research pipeline

---

### Phase 10: LLM Evaluation (DeepEval GEval / LLM-as-Judge)

> Uses [DeepEval](https://github.com/confident-ai/deepeval)'s **GEval** metric — a structured LLM-as-Judge framework where evaluation criteria are defined in plain English and scored by an LLM. More reliable than a hand-rolled judge prompt.

- [x] **10.1** Add `deepeval` to `requirements.txt`; verify it installs cleanly in the venv
- [x] **10.2** Create `src/eval/llm_eval.py` — define `GEval` metric instances (relevance, coherence, accuracy, completeness) and a `evaluate_content(test_case, metrics)` helper that runs `.measure()` and returns scores + reasons
- [x] **10.3** Define per-content-type rubrics as plain-English `criteria` strings inside each `GEval` metric (blog, LinkedIn post, research summary) — no raw judge prompts needed
- [x] **10.4** After each generation node, create a `LLMTestCase` and call `evaluate_content()` — surface the score + reason in CLI output and Streamlit UI (collapsible "📊 Eval Scores" expander)
- [x] **10.5** Bridge DeepEval scores into LangSmith — after `.measure()`, call `langsmith_client.create_feedback(run_id, key, score, comment)` using the active trace run ID captured via `RunCollectorCallbackHandler`; DeepEval does not push to LangSmith natively so this is explicit glue code
- [x] **10.5.1** *(Improvement)* Drive eval judge `model` and `threshold` from `config/services.yaml` (`eval.model`, `eval.threshold`) via `Config.eval_model` / `Config.eval_threshold` — removes hardcoded values from `llm_eval.py`; injected into `evaluate_agent_state()` from `workflow.py`
- [x] **10.5.2** *(Bug fix)* `GEval` object has no attribute `passed` in the installed DeepEval version — replaced `metric.passed` with `metric.is_successful()` (stable public API) and added a fallback of `score >= threshold` for forward compatibility
- [x] **10.5.3** *(Improvement)* Run eval in a background thread so content is returned to the UI immediately — `workflow.py` exposes `run_non_blocking_eval()` which submits `evaluate_agent_state()` to a module-level `ThreadPoolExecutor(max_workers=2)` and returns `(AgentState, Future[list[EvalScore]] | None)`; CLI keeps using blocking `run()` unchanged
- [x] **10.5.4** *(Bug fix)* Eval polling called `st.rerun()` before `_render_chat_history()`, causing Streamlit to discard all uncommitted renders — fixed by moving poll call to the **end** of `main()` after all content is rendered; `_poll_eval_future()` returns a three-state string (`"idle"` / `"running"` / `"done"`) instead of a bool; `"running"` triggers `sleep(2) + st.rerun()` to throttle polling; `"done"` triggers an immediate `st.rerun()` so eval scores appear as soon as the thread finishes — without waiting for the next user message
- [x] **10.5.5** *(Improvement)* Add `eval.enabled` boolean flag to `config/services.yaml` — when `false`, both `run()` and `run_non_blocking_eval()` skip eval entirely with no thread submitted; `Config.eval_enabled` property reads the flag (defaults `True` if absent)
- [ ] **10.6** _(Optional)_ Pairwise comparison — create two `LLMTestCase` instances (original vs refined) and use a custom `GEval` with a "which is better and why" criterion, or DeepEval's built-in `evaluate()` with both

---

### Phase 11: Observability & Monitoring Polish

- [ ] **11.1** Add custom LangSmith metadata to traces — tag by agent type, content type, user session
- [ ] **11.2** Track latency per agent node — identify slow steps in the graph
- [ ] **11.3** Track token usage and estimated cost per request via LangSmith
- [ ] **11.4** Set up LangSmith dataset + evaluator runs for regression testing content quality

---

### Phase 12: Testing *(partial[-])*

- [-] **12.1** Write unit tests for each agent (`tests/unit/`) — inject mock tools/LLM via constructor (easy to test thanks to **DIP**), test prompt construction, validate output Pydantic models (partial - linkedin_agent)
- [-] **12.2** Write unit tests for integration clients — test each implements its base interface contract, test retry logic, error handling (partial - openai_client)
- [-] **12.3** Write unit tests for LLM eval module — test scoring logic with canned inputs
- [-] **12.4** Write integration tests (`tests/integration/`) — test full agent → tool → LLM pipeline (partial - linkedin_agent)
- [-] **12.5** Write E2E tests (`tests/e2e/`) — test full workflow from user query → routed agent → generated content → eval score

---

### Phase 13: Documentation

- [x] **13.1** Write comprehensive `README.md` — overview, architecture, setup instructions, usage examples, troubleshooting
- [ ] **13.2** Write `docs/architecture.md` — system architecture decisions, LangGraph design, agent design rationale
- [ ] **13.3** Write `docs/api_documentation.md` — document each agent's input/output schema, tool configurations
- [ ] **13.4** Add inline docstrings to all public functions and classes

---

### Phase 14 (OPTIONAL): Advanced Features & Deployment

- [ ] **14.1** Add multi-turn conversation refinement — "make it more formal", "add more statistics", etc.
- [ ] **14.2** Add content series generation — create related content pieces for a campaign from one topic
- [ ] **14.3** Brand voice consistency — apply tone/style guidelines across all output
- [ ] **14.4** Content quality enhancement pipeline — post-generation optimization pass
- [ ] **14.5** SEO scoring — rate blog output on keyword density, readability, structure
- [ ] **14.6** Platform-specific formatting validation (LinkedIn char limits, blog structure)
- [ ] **14.7** Response caching layer to reduce redundant API calls and control costs
- [ ] **14.8** `docker-compose.yml` + Dockerfile for containerized deployment
- [ ] **14.9** Write `docs/deployment_guide.md` — Docker setup, env configuration, cloud deployment options
- [ ] **14.10** CMS integration — publish directly to WordPress/Ghost/Medium
- [ ] **14.11** Social media scheduling — integrate with Buffer/Hootsuite
- [ ] **14.12** Fact-checking heuristics — cross-reference claims across multiple sources

---

## Dependency Graph (Suggested Order)

```
Phase 1  (Project Setup)
  → Phase 2  (Config & LangSmith Setup)
  → Phase 3  (Integration Clients / LangChain Tools)
  → Phase 4  (LangGraph State & Workflow)
  → Phase 5  (All 6 Agents)
  → Phase 6  (Router & Orchestration)
  → Phase 7  (Context-Aware Query Enrichment)
  → Phase 8  (Streamlit UI)
  → Phase 9  (FallbackAgent & Off-Topic Handling)
  → Phase 10 (LLM Evaluation)
  → Phase 11 (Observability & Monitoring Polish)
  → Phase 12 (Testing)
  → Phase 13 (Documentation)
  → Phase 14 (Optional: Advanced Features & Deployment)
```

**Total: ~60 core tasks + ~12 optional tasks across 14 phases**
