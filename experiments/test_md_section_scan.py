from enum import Enum
from typing import cast

import mistune

# the default renderer is HTML
parse = mistune.create_markdown(renderer=None)


code_section = """
"""

list_section = """
"""


class ParagraphType(Enum):
    BLANK_LINE = "blank_line"
    THEMATIC_BREAK = "thematic_break"
    PARAGRAPH = "paragraph"
    BLOCK_CODE = "block_code"
    LIST = "list"
    TABLE = "table"


if __name__ == "__main__":
    # abstract syntax tree
    ast = cast(list, parse(list_section))

    code_types: list[str] = []
    for node in ast:
        node = cast(dict, node)
        node_type = node.get("type", "")
        code_types.append(node_type)
