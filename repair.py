import os
import glob

# 1. Replace torch.compile conditionals
for filepath in glob.glob(r'd:\@Women_safety\women_safety\core\**\*.py', recursive=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Avoid replacing if already replaced
    if 'sys.platform != "win32"' not in content and 'if hasattr(torch, "compile"):' in content:
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if line.lstrip() == 'if hasattr(torch, "compile"):':
                spaces = len(line) - len(line.lstrip())
                new_lines.append(' ' * spaces + 'import sys')
                new_lines.append(' ' * spaces + 'if hasattr(torch, "compile") and sys.platform != "win32":')
            elif line.lstrip() == 'if hasattr(torch, "compile") and sys.platform != "win32":':
                # Already replaced in another way
                new_lines.append(line)
            else:
                new_lines.append(line)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print(f"Fixed torch.compile in {filepath}")

# 2. Fix AssaultDetector `_model` to `net`
assault_path = r'd:\@Women_safety\women_safety\core\assault\detector.py'
if os.path.exists(assault_path):
    with open(assault_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'self._model' in content:
        content = content.replace('self._model', 'self.net')
        with open(assault_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed self._model -> self.net in {assault_path}")
