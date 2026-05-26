# LLM Cost Optimization for Agent Workflows: A Practical Guide

> Originally published on [omnithium.ai](https://omnithium.ai/blog/llm-cost-optimization-agents)

AI agents burn through tokens fast. A single multi-step [agent workflow](https://omnithium.ai/blog/enterprise-ai-agent-orchestration-patterns.html), classify an intent, retrieve context, reason over it, draft a response, validate the output, can easily consume 15,000-40,000 tokens per request before you've blinked. At GPT-4o pricing, that's $0.06–$0.16 per workflow execution. Run that at 50,000 executions per day and you're looking at $3,000–$8,000 _daily_ in inference alone.

That number gets attention in budget reviews.

The good news: most [production agent systems](https://omnithium.ai/blog/ai-agent-maturity-model.html) waste 40-60% of their token spend on problems that are entirely solvable with the right engineering. This guide covers the concrete techniques, token budget management, task-aware model routing, prompt caching, request batching, and output compression, that actually move the needle, along with the failure modes to watch for when applying each.

---

## Where Agent Costs Actually Come From

Before optimizing, you need an accurate cost model. Agent workflows have a different cost profile than simple chatbot completions, and the differences matter for where you invest effort.

**The four major cost drivers in agent workflows:**

1. **System prompt repetition.** Every call to an LLM re-sends the system prompt. In agentic loops with 10-20 turns, a 2,000-token system prompt gets sent 10-20 times. That's 20,000-40,000 tokens of pure repetition per workflow execution.

2. **Context accumulation.** Agents typically pass the full conversation history to maintain coherence. A workflow that starts at 500 tokens and adds 300 tokens per turn hits 3,500 tokens by turn 10, and the cost growth is superlinear because both input and output costs compound against the growing context.

3. **Model misallocation.** Using a frontier model (GPT-4o, Claude Opus, Gemini 1.5 Pro) for tasks that a smaller model handles perfectly well, intent classification, JSON extraction, format validation, is the single most common and most correctable source of waste.

4. **Speculative execution.** Many agent frameworks eagerly invoke tools and LLMs "just in case," rather than conditionally. An agent that always runs a web search step, even when the cached result from 4 minutes ago is still valid, is burning money on unnecessary compute.

A realistic cost attribution breakdown for a mid-complexity customer support agent we've analyzed looks like this:

| Cost Driver | % of Total Token Spend |
| ----------------------------------- | ---------------------- |
| System prompt repetition | 28% |
| Accumulated context window | 31% |
| Frontier model on simple tasks | 22% |
| Unnecessary tool calls / re-fetches | 12% |
| Output verbosity (over-generation) | 7% |

These proportions vary by workflow type, but across the deployments we've instrumented, the first three categories consistently dominate. That's where we'll focus.

---

## Token Budget Management

A token budget is an explicit constraint you set on how many tokens an agent can consume across its reasoning process, and it's the most underused lever available to platform engineers.

### Setting Hard and Soft Budgets

A **hard budget** aborts execution or forces summarization when a token threshold is hit. A **soft budget** triggers a warning and may switch the agent into a more economical reasoning mode, shorter outputs, fewer tool calls, compressed context.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TokenBudget:
 soft_limit: int # warn and compress
 hard_limit: int # abort or summarize
 per_turn_limit: int # max tokens per individual LLM call
 output_limit: int # max output tokens per call

 def check(self, consumed: int) -> str:
 if consumed >= self.hard_limit:
 return "hard"
 if consumed >= self.soft_limit:
 return "soft"
 return "ok"


class BudgetedAgentRunner:
 def __init__(self, budget: TokenBudget, llm_client):
 self.budget = budget
 self.llm = llm_client
 self.total_consumed = 0

 def run_turn(self, messages: list, **kwargs) -> dict:
 status = self.budget.check(self.total_consumed)

 if status == "hard":
 # Force summarization instead of continuing
 return self._summarize_and_close(messages)

 if status == "soft":
 # Switch to compressed reasoning mode
 kwargs["max_tokens"] = min(
 kwargs.get("max_tokens", self.budget.output_limit),
 self.budget.output_limit // 2,
 )
 messages = self._compress_context(messages)

 response = self.llm.complete(
 messages=messages,
 max_tokens=min(
 kwargs.get("max_tokens", self.budget.output_limit),
 self.budget.per_turn_limit,
 ),
 )

 self.total_consumed += response.usage.total_tokens
 return response

 def _compress_context(self, messages: list) -> list:
 """Keep system prompt + last N turns, summarize the middle."""
 if len(messages) <= 4:
 return messages
 system = [m for m in messages if m["role"] == "system"]
 recent = messages[-3:]
 middle = messages[len(system):-3]
 summary = self._summarize_turns(middle)
 return system + [{"role": "assistant", "content": summary}] + recent

 def _summarize_turns(self, messages: list) -> str:
 # In practice, call a cheap model (e.g., gpt-4o-mini) for this
 combined = " ".join(m["content"] for m in messages)
 return f"[Summary of prior context: {combined[:500]}...]"

 def _summarize_and_close(self, messages: list) -> dict:
 return {
 "content": "Token budget exhausted. Providing best available answer from context gathered so far.",
 "finish_reason": "budget_exceeded",
 }
```

### Context Window Pruning

Rather than passing the full message history every turn, implement rolling [context management](https://omnithium.ai/blog/memory-context-management-agents.html) that retains high-signal turns and summarizes or drops low-signal ones.

The heuristic that works well in practice: retain the system prompt, the most recent 3-5 turns unconditionally, and any turns that contain tool call results or explicit user corrections. Summarize everything else.

A more sophisticated approach uses token-weighted retention: each message gets a relevance score based on recency and information density (tool results score high; acknowledgment messages score low), and the context window is filled greedily from highest to lowest score until the budget is consumed.

```python
def score_message(message: dict, turn_index: int, total_turns: int) -> float:
 recency_score = turn_index / total_turns # higher = more recent
 content = message.get("content", "")

 type_score = 0.0
 if message.get("role") == "tool":
 type_score = 0.9 # tool results are high-value
 elif "error" in content.lower() or "correction" in content.lower():
 type_score = 0.8
 elif message.get("role") == "user":
 type_score = 0.6
 else:
 type_score = 0.3 # assistant reasoning is often reconstructable

 return 0.6 * recency_score + 0.4 * type_score


def prune_context(messages: list, token_budget: int, tokenizer) -> list:
 system_messages = [m for m in messages if m["role"] == "system"]
 non_system = [m for m in messages if m["role"] != "system"]

 scored = [
 (score_message(m, i, len(non_system)), m)
 for i, m in enumerate(non_system)
 ]
 scored.sort(key=lambda x: x[0], reverse=True)

 retained = []
 tokens_used = sum(len(tokenizer.encode(m["content"])) for m in system_messages)

 for score, message in scored:
 msg_tokens = len(tokenizer.encode(message.get("content", "")))
 if tokens_used + msg_tokens <= token_budget:
 retained.append(message)
 tokens_used += msg_tokens

 # Re-sort retained messages by original order
 original_order = {id(m): i for i, m in enumerate(non_system)}
 retained.sort(key=lambda m: original_order[id(m)])

 return system_messages + retained
```

In our testing, rolling context pruning with relevance scoring reduces input token consumption by 35-50% on workflows with more than 8 turns, with minimal measurable degradation in output quality for support, code review, and data extraction workflows.

---

## Model Routing by Task Complexity

The biggest single lever in LLM cost optimization is routing tasks to the cheapest model capable of handling them reliably. This is often called **model tiering** or **cascade routing**.

### Defining Your Model Tiers

A practical three-tier setup for most enterprise agent deployments:

| Tier | Models | Cost Range (per 1M tokens) | Suitable Tasks |
| ----------------- | ---------------------------------------- | -------------------------- | ----------------------------------------------------------------------- |
| Tier 1 (Cheap) | GPT-4o Mini, Gemini Flash, Mistral 7B | $0.10–$0.40 | Classification, extraction, formatting, validation |
| Tier 2 (Mid) | GPT-4o, Claude Haiku, Gemini 1.5 Flash | $1.00–$3.00 | Summarization, structured reasoning, code generation (simple) |
| Tier 3 (Frontier) | Claude Opus, Gemini 1.5 Pro, GPT-4o full | $10–$20 | Complex reasoning, ambiguous multi-step planning, high-stakes decisions |

### Implementing a Router

The router itself should be lightweight, ideally a Tier 1 model call or a local classifier. Using a frontier model to decide which model to use defeats the purpose.

```python
import re
from enum import Enum
from typing import Callable

class ModelTier(Enum):
 CHEAP = "gpt-4o-mini"
 MID = "gpt-4o"
 FRONTIER = "claude-opus-4"

ROUTING_RULES = [
 # Rule: (condition_fn, tier, reason)
 (lambda task: task["type"] in ("classify", "extract", "validate"), ModelTier.CHEAP, "simple structured task"),
 (lambda task: task["type"] == "summarize" and task.get("doc_length", 0) < 5000, ModelTier.CHEAP, "short summarization"),
 (lambda task: task["type"] == "summarize" and task.get("doc_length", 0) >= 5000, ModelTier.MID, "long summarization"),
 (lambda task: task["type"] == "code_review" and not task.get("security_sensitive"), ModelTier.MID, "standard code review"),
 (lambda task: task.get("requires_multi_step_reasoning"), ModelTier.FRONTIER, "complex reasoning required"),
 (lambda task: task.get("confidence_threshold", 0) > 0.95, ModelTier.FRONTIER, "high-confidence requirement"),
]

def route_task(task: dict) -> tuple[ModelTier, str]:
 for condition, tier, reason in ROUTING_RULES:
 if condition(task):
 return tier, reason
 return ModelTier.MID, "default routing" # safe default


class CascadeRouter:
 """Tries cheap model first, escalates if confidence is low."""

 def __init__(self, llm_clients: dict, confidence_threshold: float = 0.85):
 self.clients = llm_clients
 self.threshold = confidence_threshold

 def complete(self, task: dict, prompt: str) -> dict:
 tier, reason = route_task(task)

 # For tasks that might benefit from cascade, try cheap first
 if tier == ModelTier.FRONTIER and task.get("allow_cascade", False):
 cheap_result = self._try_tier(ModelTier.CHEAP, prompt)
 if cheap_result["confidence"] >= self.threshold:
 cheap_result["routing"] = "cascade_cheap_success"
 return cheap_result

 mid_result = self._try_tier(ModelTier.MID, prompt)
 if mid_result["confidence"] >= self.threshold:
 mid_result["routing"] = "cascade_mid_success"
 return mid_result

 result = self._try_tier(tier, prompt)
 result["routing"] = f"direct_{tier.value}_{reason}"
 return result

 def _try_tier(self, tier: ModelTier, prompt: str) -> dict:
 client = self.clients[tier]
 response = client.complete(prompt)
 # Confidence extraction depends on your output format;
 # for structured outputs, parse a confidence field
 confidence = self._extract_confidence(response.content)
 return {"content": response.content, "confidence": confidence, "model": tier.value}

 def _extract_confidence(self, content: str) -> float:
 # Assumes model outputs a JSON blob with a confidence key
 match = re.search(r'"confidence":\s*([0-9.]+)', content)
 return float(match.group(1)) if match else 0.75
```

### What Cascade Routing Actually Saves

In a real document processing pipeline we instrumented, applying cascade routing reduced average per-task model cost by 61%. The distribution shifted from 85% of tasks hitting the frontier model to 85% being handled by Tier 1 or Tier 2, with the frontier model reserved for genuinely complex cases. The key insight: task complexity follows a power law, and the long tail of simple tasks is where most volume, and therefore most cost, sits.

The failure mode to watch: routing rules that are too aggressive. If your Tier 1 model produces low-quality outputs on edge cases and you don't have confidence scoring in place to catch them, you'll silently degrade output quality. Always instrument output quality metrics (BLEU, human evaluation samples, downstream error rates) alongside cost metrics.

---

## Prompt Caching

Several LLM providers now offer native prompt caching: if the prefix of a request matches a cached prefix, the provider charges a reduced rate for the cached portion (typically 50-80% discount on input tokens).

**Providers with prompt caching as of 2026:**

- Anthropic: Automatic caching for prompts > 1,024 tokens; cached tokens billed at ~10% of standard rate
- OpenAI: Cached input tokens at 50% discount for prompts > 1,024 tokens
- Google Gemini: Context caching with explicit TTL management

### Structuring Prompts for Maximum Cache Hits

Prompt caching only works if the cached prefix is _identical_ across requests. This requires deliberate prompt architecture.

**The golden rule: static content first, dynamic content last.**

```python
# Bad: dynamic content interrupts the static prefix
def build_prompt_bad(user_query: str, retrieved_docs: list) -> str:
 return f"""
 You are a customer support agent. Today's date is {datetime.now()}.
 The user asked: {user_query}

 Here are the relevant docs:
 {format_docs(retrieved_docs)}

 Your guidelines are:
 [2000 tokens of static policy content here]
 """
 # Cache miss every time because date and query change the prefix

# Good: static content forms the prefix, dynamic content appended
STATIC_SYSTEM_PROMPT = """
You are a customer support agent for Acme Corp.

[2000 tokens of static policy content here]

You will receive the user query and relevant documents below.
Respond according to the guidelines above.
"""

def build_prompt_good(user_query: str, retrieved_docs: list) -> list:
 return [
 {"role": "system", "content": STATIC_SYSTEM_PROMPT}, # cacheable
 {"role": "user", "content": f"Query: {user_query}\n\nDocuments:\n{format_docs(retrieved_docs)}"}
 ]
```

For workflows where even the retrieved documents are repeated across requests (e.g., a knowledge base that doesn't change frequently), you can push the documents into the system prompt and cache the entire prefix including them:

```python
def build_cacheable_rag_prompt(static_docs: str, user_query: str) -> list:
 """
 When static_docs doesn't change often, include it in the system prompt
 so the entire prefix (system prompt + docs) is cached.
 Effective when the same doc set is reused across thousands of requests.
 """
 system = f"""{STATIC_SYSTEM_PROMPT}

## Reference Documents (current as of last refresh)
{static_docs}
"""
 return [
 {"role": "system", "content": system},
 {"role": "user", "content": user_query}
 ]
```

At scale, prompt caching on a high-volume agent workflow where the system prompt and knowledge base are static can reduce input token costs by 60-70% without any change to output quality. The catch: you need to pay attention to cache TTLs. Anthropic caches for 5 minutes by default (extendable); OpenAI's cache is automatic but has similar TTL behavior. Requests that are too infrequent won't see cache hits.

---

## Request Batching

For workflows that don't require real-time responses, document processing pipelines, nightly data enrichment, batch classification jobs, batching LLM requests reduces cost and improves throughput.

OpenAI's Batch API offers 50% cost reduction on async batch requests. Anthropic has a similar async offering. The tradeoff is latency: batch jobs complete within 24 hours rather than seconds.

```python
import json
import time
from openai import OpenAI

client = OpenAI()

def submit_batch_classification(records: list[dict]) -> str:
 """Submit a batch of classification tasks and return the batch ID."""

 requests = []
 for i, record in enumerate(records):
 requests.append({
 "custom_id": f"record-{record['id']}",
 "method": "POST",
 "url": "/v1/chat/completions",
 "body": {
 "model": "gpt-4o-mini",
 "messages": [
 {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
 {"role": "user", "content": record["text"]},
 ],
 "max_tokens": 50,
 "response_format": {"type": "json_object"},
 }
 })

 # Write requests to a JSONL file
 with open("/tmp/batch_requests.jsonl", "w") as f:
 for req in requests:
 f.write(json.dumps(req) + "\n")

 # Upload and submit
 with open("/tmp/batch_requests.jsonl", "rb") as f:
 batch_file = client.files.create(file=f, purpose="batch")

 batch = client.batches.create(
 input_file_id=batch_file.id,
 endpoint="/v1/chat/completions",
 completion_window="24h",
 )

 return batch.id


def poll_and_retrieve_batch(batch_id: str, poll_interval: int = 60) -> list[dict]:
 """Poll batch status and retrieve results when complete."""
 while True:
 batch = client.batches.retrieve(batch_id)

 if batch.status == "completed":
 output_file = client.files.content(batch.output_file_id)
 results = []
 for line in output_file.text.splitlines():
 result = json.loads(line)
 results.append({
 "id": result["custom_id"],
 "content": result["response"]["body"]["choices"][0]["message"]["content"],
 })
 return results

 if batch.status in ("failed", "cancelled", "expired"):
 raise RuntimeError(f"Batch {batch_id} ended with status: {batch.status}")

 time.sleep(poll_interval)
```

Batching is a significant optimization for the right use cases. A document enrichment pipeline processing 100,000 records daily can cut inference costs in half, from roughly $500/day to $250/day for a mid-tier model, with zero change to output quality. The constraint is real: you cannot batch anything that requires a response within seconds. Design your workflow so latency-insensitive stages are explicitly separated from real-time stages.

---

## Output Compression and Over-Generation Control

LLM outputs are frequently longer than they need to be. This is partly a training artifact, models are rewarded for helpfulness, and more words often signal effort, and partly a prompting problem. Left unconstrained, frontier models will add caveats, summaries, and alternative suggestions that the downstream system never uses.

### Prompting for Concision

The most effective technique is explicit, specific output constraints in the prompt:

```python
CONCISE_EXTRACTION_PROMPT = """
Extract the following fields from the support ticket. Return ONLY valid JSON.
No explanation, no caveats, no markdown formatting.

Fields to extract:
- customer_id (string)
- issue_category (one of: billing, technical, account, other)
- severity (one of: low, medium, high, critical)
- sentiment (one of: positive, neutral, negative, frustrated)

Return exactly this structure:
{"customer_id": "...", "issue_category": "...", "severity": "...", "sentiment": "..."}
"""
```

Constraining output format this way reduces output token consumption by 60-80% compared to open-ended prompting on structured extraction tasks. Use `max_tokens` as a hard ceiling, if you know a valid JSON response is at most 100 tokens, set `max_tokens=150` and treat any output near that limit as a signal your prompt is under-specified.

### Structured Output Modes

When the provider supports it, use structured output or JSON mode. These modes constrain the model to produce valid JSON matching a schema, which eliminates preamble text, explanatory prose, and formatting tokens:

```python
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI

client = OpenAI()

class TicketClassification(BaseModel):
 customer_id: str
 issue_category: Literal["billing", "technical", "account", "other"]
 severity: Literal["low", "medium", "high", "critical"]
 sentiment: Literal["positive", "neutral", "negative", "frustrated"]

def classify_ticket(ticket_text: str) -> TicketClassification:
 response = client.beta.chat.completions.parse(
 model="gpt-4o-mini",
 messages=[
 {"role": "system", "content": "Extract ticket metadata."},
 {"role": "user", "content": ticket_text},
 ],
 response_format=TicketClassification,
 max_tokens=100,
 )
 return response.choices[0].message.parsed
```

Structured outputs are the most reliable path to output compression for extraction and classification tasks. They also eliminate downstream parsing failures, which is a hidden cost, parsing retries consume tokens and engineering time.

---

## Putting It Together: A Cost Optimization Stack

These techniques compound. Applied together in a production agent workflow, the savings are substantial:

| Technique | Estimated Reduction |
| -------------------------------------------------- | ----------------------------------------------- |
| Token budget management + context pruning | 35-50% on input tokens for multi-turn workflows |
| Model routing (task-appropriate tiers) | 50-70% on per-task model cost |
| Prompt caching (high-volume static prefixes) | 60-70% on cached input tokens |
| Async batch processing (latency-insensitive tasks) | 50% flat via provider batch pricing |
| Output compression + structured outputs | 60-80% on output tokens for structured tasks |

These don't stack multiplicatively across the same token, they apply to different portions of your cost profile, but applied to the right workflows, it's common to see overall [inference costs drop](https://omnithium.ai/blog/measuring-ai-agent-roi.html) by 60-75% without measurable quality degradation.

### Instrumentation Is Non-Negotiable

None of this works without [observability](https://omnithium.ai/blog/agent-observability-beyond-uptime.html). You need per-workflow, per-model, per-task cost attribution to know which optimizations are having impact and which have introduced quality regressions.

At minimum, track:

- Total tokens consumed per workflow execution (input / output separately)
- Model used per task within the workflow
- Cache hit rate on prompt prefixes
- Output quality signal (downstream error rate, validation pass rate, human eval sample)
- Cost per successful execution (distinct from cost per execution, failed executions cost money too)

Build dashboards that show cost trends alongside quality metrics. An optimization that cuts costs 40% but increases error rate 15% is not a win, it's a latency bomb waiting to surface in your support queue.

---

## Failure Modes to Watch For

**Over-aggressive routing to cheap models.** Tier 1 models fail in non-obvious ways on edge cases. Always run shadow evaluations before switching high-volume tasks to cheaper models, and maintain output quality monitoring in production.

**Cache invalidation misses.** If your static prompt content changes, policy updates, new instructions, revised guidelines, and you forget to invalidate the cached prefix, you'll serve stale behavior until the TTL expires. Treat [prompt versions](https://omnithium.ai/blog/prompt-versioning-regression-testing.html) like code versions.

**Budget misconfigurations.** A token budget set too low causes agents to produce degraded, truncated answers that look fine in unit tests but fail under real query diversity. Budget values should be derived from empirical p99 consumption measurements, not guesses.

**Batch pipeline delays masking failures.** Async batch jobs that fail silently or return partial results can cause data pipeline issues downstream. Always implement explicit batch status monitoring and dead-letter queues for failed items.

**Structured output mode instability on edge cases.** JSON mode reduces output token waste but can cause model refusals or malformed output on unusual inputs. Build validation and retry logic with exponential backoff, and route persistent failures to a more capable model rather than retrying indefinitely.

---

## Conclusion

LLM cost optimization for agent workflows is an engineering discipline, not a procurement negotiation. The biggest gains come from four places: routing tasks to models that are actually capable enough, not frontier models by default, managing context accumulation aggressively, structuring prompts to maximize cache hits, and separating latency-insensitive work into batch pipelines.

Apply these in order of impact: model routing first (highest , lowest implementation risk), then context pruning (medium complexity, high impact on multi-turn workflows), then prompt caching (requires prompt architecture discipline), then batching (only viable where latency permits). Output compression is a consistent background win that should be applied everywhere.

The discipline that separates teams who sustain these savings from teams who don't is instrumentation. Cost optimization without quality monitoring is just risk transfer. Measure both, and treat any divergence between cost trends and quality trends as an incident worth investigating.

Omnithium’s platform provides built-in cost controls, model routing, and prompt caching to make these optimizations automatic, lowering your inference spend without degrading performance. Start streamlining your agent costs on [Omnithium](https://omnithium.ai) and check out [pricing](https://omnithium.ai/pricing) for your scale.

---

*Originally published on the [Omnithium Blog](https://omnithium.ai/blog/llm-cost-optimization-agents).*

📚 Explore more articles on the [Omnithium Blog](https://omnithium.ai/blog)

🚀 [Get started with Omnithium](https://omnithium.ai/signup) | [Explore the platform](https://omnithium.ai/platform/) | [Book a demo](https://omnithium.ai/demo/) | [Resources](https://omnithium.ai/resources)

---

**[Omnithium](https://omnithium.ai)** -- the AI agent platform for enterprises.

📚 [Explore the Omnithium Blog](https://omnithium.ai/blog) for more insights.

🚀 [Get started](https://omnithium.ai/signup) | [Explore the platform](https://omnithium.ai/platform/) | [Book a demo](https://omnithium.ai/demo/) | [Resources](https://omnithium.ai/resources)
