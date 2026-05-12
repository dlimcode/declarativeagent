#!/usr/bin/env bash
# Idempotently apply our project's patches to the tau2-bench clone.
#
# These patches exist because:
#   1. Prof's `scripts/run.py` registers a `deepseek_embeddings` retrieval variant
#      that points at `classic_rag_deepseek_no_grep.md`, but the template file is
#      not in the upstream tau2-bench repo.
#   2. Prof's `src/agents/local_embedder.py` imports `sentence_transformers`, but
#      tau2-bench's pyproject.toml does not declare it (or `transformers` /
#      `huggingface-hub` at compatible versions).
#
# This script is safe to run repeatedly. After it runs, do:
#   cd tau2-bench && uv sync --extra knowledge --extra voice --extra embedding

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAU2="$ROOT/tau2-bench"

if [ ! -d "$TAU2" ]; then
  echo "Error: $TAU2 not found. Clone tau2-bench first (see Makefile setup target)." >&2
  exit 1
fi

# --- Patch 1: missing prompt template -----------------------------------------
TEMPLATE_DIR="$TAU2/data/tau2/domains/banking_knowledge/prompts"
TEMPLATE="$TEMPLATE_DIR/classic_rag_deepseek_no_grep.md"

if [ ! -f "$TEMPLATE" ]; then
  if [ ! -d "$TEMPLATE_DIR" ]; then
    echo "Error: prompts dir not found: $TEMPLATE_DIR" >&2
    echo "  Did you run 'cd tau2-bench && git checkout v1.0.0'?" >&2
    exit 1
  fi
  echo "[patch] creating $TEMPLATE"
  cat > "$TEMPLATE" <<'TEMPLATE_EOF'
{{component:policy_header}}

**Search the knowledge base** for relevant information when appropriate using the provided `KB_search` tool (uses all-MiniLM-L6-v2 embeddings for retrieval).

{{component:additional_instructions}}
TEMPLATE_EOF
else
  echo "[patch] prompt template already present: $TEMPLATE"
fi

# --- Patch 2: add `embedding` extra to pyproject.toml -------------------------
PYPROJECT="$TAU2/pyproject.toml"

if grep -q '^embedding = \[' "$PYPROJECT"; then
  echo "[patch] embedding extra already in pyproject.toml"
else
  echo "[patch] adding embedding extra to pyproject.toml"
  python3 - "$PYPROJECT" <<'PY_EOF'
import re, sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

# Insert the new extra immediately after the `knowledge = [ ... ]` block.
pattern = re.compile(r'(knowledge = \[\n(?:    [^\n]*\n)*\])')
replacement = (
    r'\1\n'
    'embedding = [\n'
    '    "sentence-transformers>=3.0,<4",\n'
    '    "transformers>=4.40,<5",\n'
    '    "huggingface-hub>=0.20,<1",\n'
    ']'
)
new_content, count = pattern.subn(replacement, content, count=1)
if count != 1:
    sys.stderr.write(
        "Unable to locate `knowledge = [...]` block in pyproject.toml; "
        "tau2-bench upstream layout may have changed.\n"
    )
    sys.exit(1)

with open(path, 'w') as f:
    f.write(new_content)
PY_EOF
fi

echo "[patch] done. Now run: cd tau2-bench && uv sync --extra knowledge --extra voice --extra embedding"
