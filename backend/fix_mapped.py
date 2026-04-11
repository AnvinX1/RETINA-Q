import re

for filepath in ["app/db_models/patient.py", "app/db_models/scan.py"]:
    with open(filepath, "r") as f:
        content = f.read()
    content = re.sub(r'Optional\[Mapped\[([^\]]+)\]\]', r'Mapped[Optional[\1]]', content)
    with open(filepath, "w") as f:
        f.write(content)
print("done")
