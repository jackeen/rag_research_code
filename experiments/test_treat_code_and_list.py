""" """

import hashlib
import re

from ingestion.converters.code_block_inline import CodeTag

list_p = """
"""

code_p = """
```
"""

normal_p = """
"""

_code_tag_converter = CodeTag("granite4:3b-h")


def _get_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _process_code_blocks(text: str) -> str:
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    matches = list(pattern.finditer(text))

    tags = []
    for m in matches:
        raw_code = m.group(2).strip()
        code_id = _get_hash(raw_code)
        tag = _code_tag_converter.get_code_tag(raw_code)
        if tag is not None:
            tags.append(
                f"[CODE:{code_id}|lang={tag.lang}|desc={tag.summary.strip('.')}]"
            )

    for m, tag in zip(reversed(matches), reversed(tags)):
        text = text[: m.start()] + tag + "." + text[m.end() :]

    return text


def convert_page_with_code_block():
    print(_process_code_blocks(code_p))


if __name__ == "__main__":
    convert_page_with_code_block()
