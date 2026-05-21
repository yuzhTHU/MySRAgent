import pandas as pd
from .tag2ansi import tag2ansi

def df_to_3line(df: pd.DataFrame):
    idx = df.index
    if not isinstance(idx, pd.MultiIndex):
        idx = pd.MultiIndex.from_arrays([idx], names=df.index.names)

    rows, prev = [], None
    for i, key in enumerate(idx):
        ix = []
        for j, x in enumerate(key):
            ix.append(str(x) if prev is None or key[:j + 1] != prev[:j + 1] else "")
        rows.append(ix + [str(x) for x in df.iloc[i]])
        prev = key

    head = ["" if x is None else str(x) for x in idx.names] + [str(c) for c in df.columns]
    data = [head] + rows
    width = [max(len(r[i]) for r in data) for i in range(len(head))]
    align = ["<"] * idx.nlevels + [">" if pd.api.types.is_numeric_dtype(df[c]) else "<" for c in df.columns]
    line = tag2ansi('[gray]' + "=" * (sum(width) + 2 * (len(width) - 1)) + '[reset]')
    sep  = tag2ansi('[gray]' + "-" * (sum(width) + 2 * (len(width) - 1)) + '[reset]')

    def fmt(row, header=False):
        out = []
        for i, x in enumerate(row):
            x = f"{x:{align[i]}{width[i]}}"
            if header or i < idx.nlevels:
                x = tag2ansi('[blue]' + x + '[reset]')
            out.append(x)
        return "  ".join(out)

    lines = []
    lines.append(line)
    lines.append(fmt(head, header=True))
    lines.append(sep)
    for i, row in enumerate(rows):
        if i and idx.nlevels > 1 and idx[i][0] != idx[i - 1][0]:
            lines.append(sep)
        lines.append(fmt(row))
    lines.append(line)
    return '\n'.join(lines)