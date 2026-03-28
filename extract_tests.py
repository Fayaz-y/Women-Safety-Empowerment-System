import re

with open(r'd:\@Women_safety\ALL_SPRINTS_VALIDATION.md', 'r') as f:
    content = f.read()

# Pattern to find the filename and its python code block
pattern = r'# (tests/sprint\d_validation\.py)[\s\S]*?```python\n([\s\S]*?)```'

matches = re.findall(pattern, content)

for filename, code in matches:
    dest_path = r'd:\@Women_safety\women_safety\\' + filename
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"Written {dest_path}")
