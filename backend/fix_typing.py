import os
import re

for root, _, files in os.walk("app"):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            with open(path, "r") as fp:
                content = fp.read()
            
            if "| None" in content:
                content = re.sub(r'([a-zA-Z_0-9\[\]\.]+) \| None', r'Optional[\1]', content)
                
                if "from typing import" in content:
                    if "Optional" not in content:
                        content = content.replace("from typing import ", "from typing import Optional, ")
                else:
                    content = "from typing import Optional\n" + content
                
                with open(path, "w") as fp:
                    fp.write(content)
                print(f"Fixed {path}")

