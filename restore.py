import json

transcript = r'C:\Users\xalxo\.gemini\antigravity-ide\brain\e830b456-39b6-4168-b84e-6ecefdaabf98\.system_generated\logs\transcript_full.jsonl'
files = {}

with open(transcript, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('type') == 'PLANNER_RESPONSE':
                for tc in data.get('tool_calls', []):
                    if tc.get('name') == 'write_to_file' or tc.get('name') == 'default_api:write_to_file':
                        args = tc.get('args', {})
                        target = args.get('TargetFile')
                        content = args.get('CodeContent')
                        if target and content:
                            files[target] = content
        except Exception as e:
            pass

for target, content in files.items():
    if target.endswith('.py'):
        import emoji
        content = emoji.replace_emoji(content, replace='').replace(' \ufe0f', '').replace('\ufe0f', '')
        with open(target, 'w', encoding='utf-8') as out:
            out.write(content)
        print('Restored', target)
