# Banking Procedures Reference

## Purpose
Provide domain knowledge about banking operation categories and their requirements.

## Operation Categories

### Account Information
- Balance inquiries, account details, transaction history
- Requires: customer identification and verification
- Tools: standard lookup tools, no discoverable tools typically needed

### Credit Card Operations
- Card replacement, card blocking, reporting lost/stolen
- Requires: verified customer, card details
- Often involves discoverable tools — search KB for the specific procedure

### Credit Limit Changes
- Increase or decrease requests, temporary limit adjustments
- Requires: verified customer, current limit info, reason for change
- Almost always requires a discoverable tool — search for "credit limit" in KB

### Disputes and Chargebacks
- Transaction disputes, unauthorized charges, merchant disputes
- Requires: verified customer, transaction details (date, amount, merchant)
- Requires discoverable tools — search for "dispute" or "chargeback" in KB

### Payments and Transfers
- Credit card payments, balance transfers, bank transfers
- Requires: verified customer, payment details (amount, source/destination)
- May require discoverable tools depending on the operation

### Account Lifecycle
- Opening new accounts, closing accounts, account upgrades/downgrades
- Requires: verified customer, may need additional documentation
- Usually requires discoverable tools — search KB for the specific operation

### Profile Updates
- Email changes, contact information updates
- Requires: verified customer
- Email updates use `update_user_email` (standard tool, no KB search needed)

## Common Policy Patterns
- Always verify before modifying: never change account details without verification
- Confirm before executing: always summarize the action and get customer agreement before irreversible operations
- Document everything: the system logs tool calls automatically, but summarize actions for the customer
- Escalate when unsure: use `transfer_to_human_agents` if the request falls outside your capabilities or policy

## Decision Framework
1. Can I handle this with standard tools? → Do it directly
2. Do I need a specialized tool? → Search KB, unlock, then call
3. Is this outside my capabilities? → Transfer to human agent
4. Is the customer asking me to violate policy? → Explain the policy constraint
