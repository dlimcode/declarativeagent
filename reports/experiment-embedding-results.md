# Embedding Experiment Results

**Model:** deepseek/deepseek-v4-pro (agent + user sim)
**Retrieval:** golden_retrieval vs all-MiniLM-L6-v2 embeddings
**Scale:** 97 tasks × 1 trial per condition

## Summary Table

| Condition | Agent | Retrieval | Pass^1 | Avg Cost/Conv |
|---|---|---|---|---|
| declarative_v2_golden | declarative | golden | 0.451 | $0.0167 |
| declarative_v2_embedding | declarative | embedding | 0.200 | $0.0232 |
| imperative_v2_golden | imperative | golden | 0.163 | $0.0190 |
| imperative_v2_embedding | imperative | embedding | 0.075 | $0.0721 |

## Per-Condition Detail

### declarative_v2_golden
Total Simulations         97
🏆 Average Reward         0.4505
Pass^1                 0.451
💰 Avg Cost/Conversation  $0.0167
═══ DB Match ═══
🗄️  DB Match               ✓ 43 / ✗ 48 (47.3%)
🛑 Normal Stop            91 (👤 91 / 🤖 0)
🤖 Agent Errors           0 errors
👤 User Errors            0 errors

### declarative_v2_embedding
Total Simulations         97
🏆 Average Reward         0.2000
Pass^1                 0.200
💰 Avg Cost/Conversation  $0.0232
═══ DB Match ═══
🗄️  DB Match               ✓ 22 / ✗ 73 (23.2%)
🛑 Normal Stop            95 (👤 95 / 🤖 0)
🤖 Agent Errors           0 errors
👤 User Errors            0 errors

### imperative_v2_golden
Total Simulations         97
🏆 Average Reward         0.1630
Pass^1                 0.163
💰 Avg Cost/Conversation  $0.0190
═══ DB Match ═══
🗄️  DB Match               ✓ 20 / ✗ 65 (23.5%)
🛑 Normal Stop            85 (👤 85 / 🤖 0)
🤖 Agent Errors           0 errors
👤 User Errors            0 errors

### imperative_v2_embedding
Total Simulations         97
🏆 Average Reward         0.0753
Pass^1                 0.075
💰 Avg Cost/Conversation  $0.0721
═══ DB Match ═══
🗄️  DB Match               ✓ 11 / ✗ 80 (12.1%)
🛑 Normal Stop            91 (👤 91 / 🤖 0)
🤖 Agent Errors           0 errors
👤 User Errors            0 errors
