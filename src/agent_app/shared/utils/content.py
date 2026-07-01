import re

def remove_think_blocks(text: str) -> str:
    return re.sub(
        r"<think>\s*.*?\s*</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()

def extract_json(content: str) -> str:
    content = content.strip()

    # remove ```json ... ```
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)

    if match:
        return match.group(1)

    return content


