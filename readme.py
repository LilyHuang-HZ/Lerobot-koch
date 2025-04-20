from pathlib import Path
import re

def extract_headers(md_path):
    headers = []
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("## "):
                title = line[3:].strip()
                anchor = re.sub(r'[^\w\- ]', '', title).lower().replace(" ", "-")
                headers.append((title, anchor))
    return headers

# 忽略 README 本身
md_files = [f for f in Path(".").glob("*.md") if f.name.lower() != "readme.md"]

readme_lines = [
    "# 📚 LeRobot-Koch 文档导航\n",
    "以下是自动从 markdown 文件中提取的标题目录。\n"
]

for md in sorted(md_files):
    readme_lines.append(f"\n## 📄 [{md.name}]({md.name})\n")
    for title, anchor in extract_headers(md):
        readme_lines.append(f"- [{title}]({md.name}#{anchor})")

# 写入 README.md
with open("README.md", "w", encoding="utf-8") as f:
    f.write("\n".join(readme_lines))

print("✅ README.md 自动更新完成！")
