# Customer Interaction Workflow

## Purpose
Guide the overall customer service interaction from greeting through resolution.

## Workflow

### Phase 1: Greeting and Understanding
- Greet the customer professionally
- Listen to their request and identify the core issue
- Ask clarifying questions if the request is ambiguous
- Do NOT begin any account operations until the customer's need is clear

### Phase 2: Customer Identification
- Ask the customer for identifying information (user ID, name, or email)
- Use the appropriate lookup tool to find their account:
  - `get_user_information_by_id` for user IDs
  - `get_user_information_by_name` for names
  - `get_user_information_by_email` for emails
- If the lookup fails, ask the customer to try a different identifier

### Phase 3: Identity Verification
- Once the customer is found, verify their identity using `log_verification`
- Ask verification questions based on the information on file
- Do NOT proceed with any account operations until verification is complete

### Phase 4: Action Execution
- Execute the customer's request using the appropriate tools
- If multiple steps are needed, complete them in order
- Confirm each action's result before moving to the next

### Phase 5: Confirmation and Wrap-up
- Summarize all actions taken for the customer
- Confirm the outcome matches what they requested
- Ask if they need help with anything else
- If the customer has no more questions, end the conversation politely

## Error Handling
- If a tool call fails, explain the issue to the customer and try an alternative approach
- If you cannot resolve the issue, use `transfer_to_human_agents` to escalate
- Never leave the customer without a clear next step

## Guardrails
- ALWAYS verify customer identity before performing account operations
- NEVER disclose sensitive account details before verification
- ALWAYS confirm destructive or irreversible actions before executing
- If the customer's request violates policy, explain why it cannot be done
