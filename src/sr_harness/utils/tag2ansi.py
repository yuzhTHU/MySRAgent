# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import re

__all__ = ["tag2ansi"]

RESET_CODE = {"reset": "\033[0m"}

STYLE_TABLE = {
    "bold": "1",
    "dim": "2",
    "italic": "3",
    "underline": "4",
    "blink": "5",
    "reverse": "7",
    "hidden": "8",
    "strike": "9",
}

COLOR_TABLE = {
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "brightblack": "90",
    "brightred": "91",
    "brightgreen": "92",
    "brightyellow": "93",
    "brightblue": "94",
    "brightmagenta": "95",
    "brightcyan": "96",
    "brightwhite": "97",
    # --- 扩展常用颜色 (256色或TrueColor) ---
    "orange": "38;5;208",
    "pink": "38;5;213",
    "hotpink": "38;5;205",
    "purple": "38;5;129",
    "violet": "38;5;177",
    "brown": "38;5;94",
    "olive": "38;5;100",
    "lime": "38;5;118",
    "teal": "38;5;30",
    "navy": "38;5;19",
    "gold": "38;5;220",
    "silver": "38;5;250",
    "gray": "38;5;245",
    "darkgray": "38;5;240",
    "lightgray": "38;5;252",
    "indigo": "38;5;57",
    "coral": "38;5;203",
    "skyblue": "38;5;117",
    "turquoise": "38;5;45",
}


def parse_tag(tag: str) -> str:
    """
    支持 [bold+red], [italic+#FFAA00], [color123]
    """
    parts = tag.split(" ")
    seq = ""
    for part in parts:
        part = part.lower()
        if part in STYLE_TABLE:
            seq += f"\033[{STYLE_TABLE[part]}m"
        elif part in COLOR_TABLE:
            seq += f"\033[{COLOR_TABLE[part]}m"
        elif part.startswith("color"):
            try:
                idx = int(part.removeprefix("color"))
                assert 0 <= idx <= 255
                seq += f"\033[38;5;{idx}m"
            except ValueError:
                pass
        elif part.startswith("#") and len(part) == 7:
            try:
                r = int(part[1:3], 16)
                g = int(part[3:5], 16)
                b = int(part[5:7], 16)
                seq += f"\033[38;2;{r};{g};{b}m"
            except ValueError:
                pass
        else:
            pass
    return seq


def find_matching_paren(s: str, start_idx: int) -> int:
    stack = []
    pairs = {")": "(", "}": "{", "]": "["}
    opens = set(pairs.values())
    closes = set(pairs.keys())
    for i, ch in enumerate(s[start_idx:], start=start_idx):
        if ch in opens:
            stack.append(ch)
        elif ch in closes and stack and pairs[ch] == stack[-1]:
            stack.pop()
            if not stack:
                return i
    return -1


def auto_insert_reset(segment: str) -> str:
    if (reset_pos := segment.find("[reset]")) != -1:
        return segment[:reset_pos] + RESET_CODE["reset"] + segment[reset_pos + len("[reset]") :]
    elif not (stripped := segment.lstrip()):
        return segment + RESET_CODE["reset"]
    elif (first_char := stripped[0]).isalnum():
        m = re.search(r"[^a-zA-Z0-9/_-]", stripped)
        if not m:
            return segment + RESET_CODE["reset"]
        offset = len(segment) - len(stripped)
        insert_at = offset + m.start()
        return segment[:insert_at] + RESET_CODE["reset"] + segment[insert_at:]
    elif first_char in "({[":
        offset = len(segment) - len(stripped)
        match_idx = find_matching_paren(segment, offset)
        if match_idx == -1:
            return segment + RESET_CODE["reset"]
        return segment[: match_idx + 1] + RESET_CODE["reset"] + segment[match_idx + 1 :]
    else:
        return segment + RESET_CODE["reset"]


def tag2ansi(text: str) -> str:
    """
    将 [tag1 tag2] 转换为 ANSI 码并智能插入 reset。
    支持多种颜色定义方式、样式组合、自动闭合。
    """
    pattern = re.compile(r"\[(?!reset)([a-zA-Z0-9+#]+(?: [a-zA-Z0-9+#]+)*)\]")

    result = []
    last_end = 0
    active_tag = None

    for match in pattern.finditer(text):
        tag = match.group(1)
        start, end = match.span()

        # 先加前一段
        if last_end < start:
            seg = text[last_end:start]
            if active_tag:
                seg = auto_insert_reset(seg)
            result.append(seg)
        last_end = end

        # 当前标签
        if tag.lower() == "reset":
            result.append(RESET_CODE)
            active_tag = None
        else:
            seq = parse_tag(tag)
            if seq:
                result.append(seq)
                active_tag = tag
            else:
                result.append(match.group(0))

    # 收尾
    if last_end < len(text):
        seg = text[last_end:]
        if active_tag:
            seg = auto_insert_reset(seg)
        result.append(seg)

    return "".join(result)
