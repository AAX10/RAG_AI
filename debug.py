import json

transcript = r'C:\Users\xalxo\.gemini\antigravity-ide\brain\e830b456-39b6-4168-b84e-6ecefdaabf98\.system_generated\logs\transcript_full.jsonl'
lines_read = 0
planner = 0
write_calls = 0

with open(transcript, 'r', encoding='utf-8') as f:
    for line in f:
        lines_read += 1
        data = json.loads(line)
        if data.get('type') == 'PLANNER_RESPONSE':
            planner += 1
            for tc in data.get('tool_calls', []):
                if tc.get('name') == 'default_api:write_to_file':
                    write_calls += 1

print(f"Lines: {lines_read}, Planner: {planner}, Write Calls: {write_calls}")
