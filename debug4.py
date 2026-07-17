import json

transcript = r'C:\Users\xalxo\.gemini\antigravity-ide\brain\e830b456-39b6-4168-b84e-6ecefdaabf98\.system_generated\logs\transcript_full.jsonl'

with open(transcript, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'PLANNER_RESPONSE':
            for tc in data.get('tool_calls', []):
                if tc.get('name') == 'write_to_file':
                    print("Keys in tc:", list(tc.keys()))
                    if 'arguments' in tc:
                        print("Type of arguments:", type(tc['arguments']))
                    break
