# Advanced usage

Power-user features beyond the common CLI workflow in the [README](../README.md#usage).

## OpenAI-compatible embeddings endpoint

RAGdoll can serve as a **drop-in local embedding provider** for anything that speaks the OpenAI embeddings API: Continue.dev, LangChain, LlamaIndex, Copilot extensions, custom scripts. Your text never leaves the machine.

```bash
ragdoll serve        # starts daemon on http://localhost:7474
```

```bash
# Works with any OpenAI SDK:
curl -s http://localhost:7474/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input": "how does auth work", "model": "ragdoll"}'
```

Example response:
```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.012, -0.043, ...]}],
  "model": "ragdoll",
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

**Continue.dev** `config.json`:
```json
"embeddingsProvider": {
  "provider": "openai",
  "model": "ragdoll",
  "apiBase": "http://localhost:7474/v1"
}
```

**LangChain / LlamaIndex**: point `OpenAIEmbeddings(base_url="http://localhost:7474/v1", api_key="ignored")`. No API key required, the server is local-only.

## Multiple profiles

Keep separate indexes for different workspaces by setting `RAGDOLL_DB`:

```bash
RAGDOLL_DB=~/.ragdoll/work.db ragdoll index ~/repos/work
RAGDOLL_DB=~/.ragdoll/side.db ragdoll index ~/repos/side-project
```

All commands honour the variable, so you can set it per-shell or per-project.

## Debug queries with `ragdoll explain`

Want to know why a result ranked where it did?

```bash
ragdoll explain "jwt validation"
```

Shows the per-result `vec_rank`, `bm25_rank`, and final `rrf` score. Useful for tuning queries or catching FTS misses.

## Backup and share indexes

```bash
ragdoll export index.jsonl          # dump everything (content + vectors)
ragdoll import index.jsonl --replace  # seed a fresh install
```

Handy for backups or onboarding a teammate without re-indexing from scratch.

## Always-on daemon (macOS)

Tired of restarting `ragdoll serve`? Install a launchd agent:

```bash
ragdoll autostart install --watch ~/repos/work --watch ~/repos/side
```

The daemon now starts at login, survives reboots, and logs to `~/.ragdoll/ragdoll.log`. Uninstall with `ragdoll autostart uninstall`.
