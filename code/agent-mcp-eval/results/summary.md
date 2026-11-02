| runtime | policy | pass | recall | approval | kb | flaky | multi | failures by cause | unapproved proposed / executed |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| loop/baseline | eager/naive | 5/40 | 0/8 | 0/8 | 5/8 | 0/8 | 0/8 | {'unapproved_action': 12, 'conversation_state': 12, 'retrieval': 3, 'tool_response': 8} | 12 / 12 |
| groupchat/baseline | eager/naive | 13/40 | 0/8 | 0/8 | 5/8 | 0/8 | 8/8 | {'unapproved_action': 12, 'conversation_state': 4, 'retrieval': 3, 'tool_response': 8} | 12 / 12 |
| loop/+entity_memory | eager/naive | 5/40 | 0/8 | 0/8 | 5/8 | 0/8 | 0/8 | {'unapproved_action': 16, 'retrieval': 3, 'tool_response': 8, 'conversation_state': 8} | 16 / 16 |
| loop/+safe_state | eager/naive | 13/40 | 0/8 | 0/8 | 5/8 | 0/8 | 8/8 | {'unapproved_action': 12, 'conversation_state': 4, 'retrieval': 3, 'tool_response': 8} | 12 / 12 |
| loop/+approval_gate | eager/naive | 13/40 | 4/8 | 4/8 | 5/8 | 0/8 | 0/8 | {'conversation_state': 12, 'unapproved_action': 4, 'retrieval': 3, 'tool_response': 8} | 12 / 0 |
| loop/all | eager/naive | 25/40 | 8/8 | 4/8 | 5/8 | 0/8 | 8/8 | {'unapproved_action': 4, 'retrieval': 3, 'tool_response': 8} | 16 / 0 |
| groupchat/all | eager/naive | 25/40 | 8/8 | 4/8 | 5/8 | 0/8 | 8/8 | {'unapproved_action': 4, 'retrieval': 3, 'tool_response': 8} | 16 / 0 |
| loop/baseline+careful | careful/retry | 25/40 | 4/8 | 8/8 | 5/8 | 8/8 | 0/8 | {'conversation_state': 12, 'retrieval': 3} | 0 / 0 |
| groupchat/baseline+careful | careful/retry | 33/40 | 4/8 | 8/8 | 5/8 | 8/8 | 8/8 | {'conversation_state': 4, 'retrieval': 3} | 0 / 0 |
