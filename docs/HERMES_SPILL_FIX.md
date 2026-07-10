# Hermes skill-catalog spill on abliterated DSV4

## What you saw
```
Hey visualizations.
  - mermaid: ...
  - music: ...
```
That text is **not random garbage** — it is the Hermes **skills index** being echoed in the model’s first assistant message.

## Root cause (confirmed from session 20260709_212050)
1. User: `hello`
2. Model turn 1: dumps skill catalog as **text** + calls `skill_view("hermes-agent")` (~2.6k out tokens, 89s)
3. Tool returns 53k chars of skill body
4. Model turn 2: short good greeting (~143 chars)

Hermes **shows turn-1 text**, so you see the spill even though the final turn is fine.

Drivers:
- Hermes skills prompt says “scan skills / err on the side of loading”
- Abliterated DSV4 is weaker at “don’t restate system prompt” + over-triggers `skill_view` on greetings
- Previously also: `max_tokens` = full context (65536) while serve was 64k → HTTP 400 chaos (now 1M + `model.max_tokens: 8192`)

## Fixes applied
- `model.max_tokens: 8192` (do not request full ctx as output)
- `model.extra_body.temperature: 0.0`
- `agent.environment_hint`: no skill-list echo; greetings = one short sentence, no tools

## What you must do
**Fully quit Hermes and start a fresh session** (`/new` or restart CLI) so config reloads.

```bash
# confirm API is 1M uncensored
curl -s http://10.100.10.3:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["max_model_len"])'
# should print 1048576
```

Then:
```text
hermes   # or your usual launch
/new
hello
```

Expected: one short “Hey! How can I help?” **without** mermaid/music list.

## If spill persists
1. `/new` session (old history may contain the bad assistant message)
2. Temporarily: `agent.tool_use_enforcement: false` or disable skill tools for chat
3. Or use stock (non-abliterated) weights for Hermes agenting; keep abliterated for freeform chat
