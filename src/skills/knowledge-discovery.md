# Knowledge Base and Tool Discovery

## Purpose
Guide the process of finding relevant procedures and discovering specialized tools.

## When to Use
- When the customer's request requires a procedure or tool you don't have direct access to
- When you need to look up specific policies, procedures, or tool names

## Workflow

### Step 1: Determine if KB Search is Needed
- If the relevant procedure information and tool names are already provided in your system prompt (e.g., under golden retrieval), use them directly — skip to Step 3
- If you have the `KB_search` tool available, proceed to Step 2
- If neither: use only the standard tools available to you

### Step 2: Search the Knowledge Base
- Use `KB_search` to find the procedure relevant to the customer's request
- Use descriptive search queries related to the customer's issue (e.g., "credit limit increase", "dispute transaction", "card replacement")
- Read the search results carefully — they contain procedure details and tool names
- If the results are not relevant, refine your query with different terms or synonyms
- You may search multiple times with different queries if needed

### Step 3: Extract and Use Discoverable Tools
- Look for tool names in the format `tool_name_NNNN` (e.g., `submit_credit_limit_increase_request_7392`)
- The exact name including the number suffix is required — do not guess or modify it
- To use a discoverable tool, follow this exact sequence:
  1. **Unlock**: Call `unlock_discoverable_agent_tool` with `agent_tool_name` set to the exact tool name
  2. **Call**: Call `call_discoverable_agent_tool` with `agent_tool_name` and `arguments` (a JSON string of the tool's parameters)
- Read the unlock response — it describes the tool's parameters and what arguments it expects

## Error Handling
- If `KB_search` returns no useful results after 2-3 attempts, explain to the customer that you need to transfer them to a specialist
- If `unlock_discoverable_agent_tool` fails, the tool name may be wrong — re-check the KB search results
- If `call_discoverable_agent_tool` fails, check the arguments format — it must be a valid JSON string

## Guardrails
- NEVER guess tool names — they contain random suffixes that must be exact
- ALWAYS read the full KB search results before attempting to unlock a tool
- The `arguments` parameter for `call_discoverable_agent_tool` must be a JSON string, not a dict
