| framework | config | policy | pass | recall | approval | kb | flaky | multi | failures by cause | unapproved proposed / executed |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| autogen | baseline | eager/naive | 5/40 | 0/8 | 0/8 | 5/8 | 0/8 | 0/8 | {'unapproved_action': 12, 'conversation_state': 12, 'retrieval': 3, 'tool_response': 8} | 12 / 12 |
| autogen | +safe_state | eager/naive | 13/40 | 0/8 | 0/8 | 5/8 | 0/8 | 8/8 | {'unapproved_action': 12, 'conversation_state': 4, 'retrieval': 3, 'tool_response': 8} | 12 / 12 |
| autogen | all mitigations | eager/naive | 25/40 | 8/8 | 4/8 | 5/8 | 0/8 | 8/8 | {'unapproved_action': 4, 'retrieval': 3, 'tool_response': 8} | 16 / 0 |
| autogen | baseline, careful policy | careful/retry | 25/40 | 4/8 | 8/8 | 5/8 | 8/8 | 0/8 | {'conversation_state': 12, 'retrieval': 3} | 0 / 0 |
| llamaindex | baseline | eager/naive | 5/40 | 0/8 | 0/8 | 5/8 | 0/8 | 0/8 | {'unapproved_action': 12, 'conversation_state': 12, 'retrieval': 3, 'tool_response': 8} | 12 / 12 |
| llamaindex | +safe_state | eager/naive | 13/40 | 0/8 | 0/8 | 5/8 | 0/8 | 8/8 | {'unapproved_action': 12, 'conversation_state': 4, 'retrieval': 3, 'tool_response': 8} | 12 / 12 |
| llamaindex | all mitigations | eager/naive | 25/40 | 8/8 | 4/8 | 5/8 | 0/8 | 8/8 | {'unapproved_action': 4, 'retrieval': 3, 'tool_response': 8} | 16 / 0 |
| llamaindex | baseline, careful policy | careful/retry | 25/40 | 4/8 | 8/8 | 5/8 | 8/8 | 0/8 | {'conversation_state': 12, 'retrieval': 3} | 0 / 0 |
