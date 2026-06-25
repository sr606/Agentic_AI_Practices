# Agent-to-Agent Protocol (A2A) & ACP
## Presentation Slides — 30 minutes

---

## SLIDE 1: Title Slide

**Agent-to-Agent Communication Protocols**
A2A vs ACP — When Agents Talk to Each Other

Presented by: [Your Name]
Date: [Date]

---

## SLIDE 2: Agenda

1. Why do agents need to talk to each other?
2. What is A2A (Agent-to-Agent Protocol)?
3. What is ACP (Agent Communication Protocol)?
4. A2A vs ACP — Key Differences
5. Architecture Comparison
6. Live Demo
7. When to use which?
8. Q&A

---

## SLIDE 3: The Problem — Why Agent Communication?

- Single agents are powerful, but limited in scope
- Real-world tasks often need **multiple specialized agents** working together
- Example: Planning a trip needs weather info + activity suggestions + booking
- Challenge: How do these agents discover, communicate, and coordinate with each other?
- We need **standardized protocols** — just like HTTP standardized web communication

---

## SLIDE 4: The Landscape

| Protocol | Backed By | Released | Focus |
|----------|-----------|----------|-------|
| A2A | Google | April 2025 | Agent-to-agent interoperability |
| ACP | IBM (BeeAI) | 2025 | Agent communication within platforms |
| MCP | Anthropic | Nov 2024 | Model-to-tool connectivity |

- MCP = How an agent uses tools (like a USB port)
- A2A = How agents talk to each other (like email between people)
- ACP = How agents communicate within a hosting platform (like intercom in a building)

---

## SLIDE 5: What is A2A?

**Agent-to-Agent Protocol (by Google)**

- Open protocol for agents built by different vendors to communicate
- Each agent is an **independent service** with its own endpoint
- Uses **structured message envelopes** with routing metadata
- Supports: sender, receiver, session_id, message_id, in_reply_to
- Designed for **cross-organization** interoperability

Key Idea: "Every agent is a citizen on the network with its own address"

---

## SLIDE 6: A2A — Core Concepts

1. **Agent Card** — JSON metadata describing what an agent can do (like a business card)
2. **Envelope** — Structured wrapper around every message
   - protocol, message_id, session_id
   - sender, receiver
   - type (invoke / result / error)
   - body (the actual payload)
3. **Session** — Groups related messages together
4. **Tasks** — Units of work that agents can assign to each other

---

## SLIDE 7: A2A — Message Envelope Structure

```json
{
  "protocol": "simple-a2a/1.0",
  "message_id": "uuid-here",
  "session_id": "session-uuid",
  "timestamp": "2025-06-25T10:00:00Z",
  "sender": "orchestrator",
  "receiver": "weather_agent",
  "type": "invoke",
  "body": { "city": "goa" }
}
```

- Every message is self-describing
- You always know WHO sent it, WHO it's for, and WHAT session it belongs to
- Enables tracing, replay, and debugging

---

## SLIDE 8: A2A — Architecture (Our Demo)

```
[Streamlit UI :6090]
        |
        v
[Orchestrator :6003]
      /           \
     v             v
[Weather Agent   [Activity Agent
    :6001]           :6002]
```

- 3 separate services, 3 separate ports
- Orchestrator coordinates the flow
- Each agent independently deployable & scalable
- Communication via HTTP POST with envelopes

---

## SLIDE 9: What is ACP?

**Agent Communication Protocol (by IBM / BeeAI)**

- Protocol for agents hosted on the **same platform/server**
- One server exposes multiple agents via a unified API
- RESTful design: discover agents, create runs, get results
- Uses **Messages with Parts** (multi-modal content support)
- Designed for **platform-internal** agent orchestration

Key Idea: "One platform, many agents, one API to rule them all"

---

## SLIDE 10: ACP — Core Concepts

1. **Agent Manifest** — Describes an agent (name, description, supported content types)
2. **Message** — Contains role + list of Parts
3. **MessagePart** — Content with a content_type (text/plain, application/json, image/png)
4. **Run** — A single invocation of an agent; has status, input, output, timing
5. **Discovery** — GET /agents returns all available agents

---

## SLIDE 11: ACP — API Shape

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /agents | GET | Discover all agents (manifests) |
| /agents/{name} | GET | Get a single agent's manifest |
| /runs | POST | Invoke an agent (create a Run) |
| /runs/{run_id} | GET | Get a stored Run result |

Simple, RESTful, familiar to any developer.

---

## SLIDE 12: ACP — Run Request & Response

**Request:**
```json
{
  "agent_name": "weather",
  "input": [{
    "role": "user",
    "parts": [{"content_type": "text/plain", "content": "Goa"}]
  }]
}
```

**Response:**
```json
{
  "run_id": "uuid",
  "status": "completed",
  "output": [{
    "role": "agent",
    "parts": [
      {"content_type": "text/plain", "content": "Goa is Sunny at 32C..."},
      {"content_type": "application/json", "content": "{...}"}
    ]
  }]
}
```

---

## SLIDE 13: ACP — Architecture (Our Demo)

```
[Streamlit UI :7090]
        |
        v
[ACP Server :7001]
  ┌─────────────────────────────┐
  │  trip_planner (router)      │
  │  weather agent              │
  │  activity agent             │
  └─────────────────────────────┘
```

- Single server, single port
- Router agent calls other agents via the SAME ACP API (POST /runs)
- Agent-to-agent calls happen INSIDE the platform
- All discovery via GET /agents

---

## SLIDE 14: A2A vs ACP — The Big Picture

| Aspect | A2A | ACP |
|--------|-----|-----|
| Philosophy | Agents as independent services | Agents within a shared platform |
| Deployment | Each agent = separate service | All agents = one server |
| Discovery | Agent Cards (well-known URL) | GET /agents endpoint |
| Message Format | Envelope (sender/receiver/session) | Message with Parts (role + content_type) |
| Routing | Explicit sender/receiver fields | Agent name in Run request |
| Multi-modal | Via body content | Via MessagePart content_type |
| Session Tracking | Built-in (session_id) | Via run_id |

---

## SLIDE 15: A2A vs ACP — Communication Style

**A2A: Like sending letters**
- Each message has FROM, TO, and a tracking number
- Messages carry full context (who, what, when, in-reply-to)
- Agents don't need to be on the same server
- Explicit orchestration needed

**ACP: Like calling a function on a platform**
- "Run this agent with this input"
- Platform handles routing internally
- Simpler mental model, less boilerplate
- Agents can still call other agents (router pattern)

---

## SLIDE 16: A2A vs ACP — Deployment & Scaling

**A2A:**
- Each agent is independently deployable
- Can scale individual agents differently (e.g., weather agent gets 10x traffic)
- Cross-team / cross-org friendly
- More infrastructure overhead (service discovery, load balancing, network)

**ACP:**
- Single deployment unit
- Simpler ops: one server to monitor, one health check
- Agents share resources (memory, CPU)
- Easier to get started, harder to scale individually

---

## SLIDE 17: A2A vs ACP — Traceability

**A2A:**
- Every envelope has message_id, session_id, in_reply_to
- Full conversation history is traceable across services
- Natural audit trail
- Great for compliance-heavy environments

**ACP:**
- Run ID tracks each invocation
- Router agents can expose internal hop traces
- Simpler for debugging (everything in one process)
- But cross-agent tracing requires custom implementation

---

## SLIDE 18: A2A vs ACP — Code Comparison

**A2A Agent (weather):**
```python
@app.post("/run")
def run_agent(req: WeatherRequest):
    envelope = req.envelope
    city = envelope["body"]["city"]
    # ... do work ...
    return {
        "envelope": {
            "sender": "weather_agent",
            "receiver": envelope["sender"],
            "in_reply_to": envelope["message_id"],
            "body": result
        }
    }
```

**ACP Agent (weather):**
```python
def weather_agent(text: str) -> List[MessagePart]:
    city = text.strip().lower()
    # ... do work ...
    return [MessagePart(content="Goa is Sunny...")]
```

ACP = simpler agent code. A2A = more context in every message.

---

## SLIDE 19: A2A vs ACP — When to Use Which?

**Choose A2A when:**
- Agents are built by different teams/organizations
- Agents need to be independently deployed and scaled
- You need cross-platform interoperability
- Compliance requires full message audit trails
- You're building a marketplace of agents

**Choose ACP when:**
- All agents are part of one platform/product
- You want simplicity and fast development
- Agents share the same runtime/infrastructure
- You're prototyping or building internal tools
- Multi-modal content support is a priority

---

## SLIDE 20: Real-World Analogy

**A2A = Email between companies**
- Each company has its own mail server
- Messages have headers (From, To, Subject, In-Reply-To)
- Works across organizations
- More formal, more overhead

**ACP = Slack channels within a company**
- One platform hosts all communication
- You just @mention an agent and get a response
- Fast, simple, less ceremony
- Best for internal collaboration

---

## SLIDE 21: Can They Work Together?

YES! They solve different layers of the problem.

```
┌──────────────────────────────────────┐
│  Organization A        Organization B │
│  ┌─────────────┐     ┌─────────────┐ │
│  │ ACP Platform│     │ ACP Platform│ │
│  │  Agent 1    │────▶│  Agent X    │ │
│  │  Agent 2    │ A2A │  Agent Y    │ │
│  └─────────────┘     └─────────────┘ │
└──────────────────────────────────────┘
```

- ACP for internal agent communication within a platform
- A2A for cross-platform/cross-org agent communication
- Like intranet (ACP) vs internet (A2A)

---

## SLIDE 22: The MCP + A2A + ACP Stack

```
┌─────────────────────────────────────┐
│  A2A Layer (inter-agent)            │  Agents talking to agents
├─────────────────────────────────────┤
│  ACP Layer (intra-platform)         │  Agents within a platform
├─────────────────────────────────────┤
│  MCP Layer (model-to-tools)         │  Agent using tools/APIs
└─────────────────────────────────────┘
```

- MCP: Agent connects to databases, APIs, file systems
- ACP: Multiple agents coordinate within one platform
- A2A: Agents across different platforms/vendors communicate

---

## SLIDE 23: Our Demo — What We Built

**Same use case, two protocols:**

"Give me a city → get weather → suggest activities"

| | A2A Demo | ACP Demo |
|--|----------|----------|
| Services | 3 separate (ports 6001-6003) | 1 server (port 7001) |
| Frontend | Streamlit :6090 | Streamlit :7090 |
| Agents | weather, activity, orchestrator | weather, activity, trip_planner |
| Communication | Envelope-based POST /run | ACP POST /runs |
| Trace | A2A message trace with envelopes | Router hop trace |

---

## SLIDE 24: Demo — What to Notice

When I show the demo, pay attention to:

1. **A2A Demo:**
   - 3 separate processes starting up
   - Full envelope in the trace (sender, receiver, session, message_id)
   - Each agent is a standalone FastAPI app
   - Orchestrator explicitly routes between agents

2. **ACP Demo:**
   - 1 server with all agents registered
   - Discovery endpoint (GET /agents) shows all available agents
   - Router agent uses the SAME ACP API to call other agents
   - MessageParts with different content_types (text + JSON)

---

## SLIDE 25: Key Takeaways

1. **A2A** = independent agents communicating across boundaries (Google's vision for the open agent web)
2. **ACP** = agents communicating within a shared platform (IBM's vision for agent hosting)
3. Both are **complementary**, not competing
4. Choose based on: deployment model, team boundaries, scale requirements
5. The future is **multi-protocol** — like we use HTTP, WebSocket, and gRPC together today

---

## SLIDE 26: What's Next in This Space?

- Google A2A: Growing ecosystem, enterprise adoption in progress
- IBM ACP / BeeAI: Open-source platform for hosting multi-agent systems
- Anthropic MCP: Rapidly adopted for tool integration
- Industry moving toward **interoperable agent ecosystems**
- Expect convergence and bridge protocols in 2025-2026

---

## SLIDE 27: Resources & Links

- A2A Protocol: github.com/google/A2A
- ACP Spec: github.com/i-am-bee/acp
- MCP Spec: modelcontextprotocol.io
- Our demos: [your repo links]

---

## SLIDE 28: Thank You + Q&A

Questions?

---

## BONUS: Demo Script (for your 15-minute demo)

### A2A Demo (7 minutes):
1. Show the code structure — 3 separate files, 3 separate services
2. Run `python run_all.py` — point out 3 ports starting
3. Open Streamlit UI at :6090
4. Pick "Goa" — show the result
5. Expand the A2A Message Trace — walk through each envelope
6. Highlight: sender/receiver fields, session_id consistency, in_reply_to linking

### ACP Demo (7 minutes):
1. Show the code structure — 1 server.py with all agents
2. Run `python run_all.py` — point out single port :7001
3. Open browser to localhost:7001/agents — show discovery
4. Open Streamlit UI at :7090
5. Select "trip_planner" agent, type "Manali"
6. Show the result — multiple MessageParts (text + JSON)
7. Show the router trace — agent-to-agent hops WITHIN ACP
8. Switch to "weather" agent directly — show it works standalone too

### Wrap-up (1 minute):
- Side-by-side: 3 services vs 1 server, envelopes vs runs
- "Same outcome, different architectural philosophy"
