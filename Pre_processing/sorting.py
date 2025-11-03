# sort_rows_or_transpose.py — keep row-major; transpose if column-major (no numpy/regex)
# Output preserves the exact numeric formatting from input (no re-formatting).

INPUT_FILE  = r"C:\Git\Algoim_mimic\Pre_processing\text_data\100kTestBernstein_p1_output_8_filtered64.txt"
OUTPUT_FILE = r"C:\Git\Algoim_mimic\Pre_processing\text_data\100kTestBernstein_p1_output_8_filtered64_sorted.txt"

PER_LINE     = 8        # number of nodes per row (columns)
STRICT_GRID  = True     # require n % PER_LINE == 0 else skip record

# ---------- helpers (token-preserving parsing) ----------

def parse_tokens_and_floats(text):
    """
    Return (tokens, floats) for a comma-separated list.
    Tokens are the original substrings (trimmed) to preserve formatting in output.
    Floats are used for geometry-driven ordering decisions.
    Brackets [ ... ] are tolerated and ignored if present.
    """
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    tokens = [t.strip() for t in s.split(",") if t.strip() != ""]
    floats = [float(t) for t in tokens]
    return tokens, floats

def _largest_gaps_partition(sorted_idx, values, k_groups):
    """
    Indices 'sorted_idx' must already be sorted by 'values' ascending.
    Split into k_groups by cutting at the (k_groups-1) largest consecutive gaps in 'values'.
    """
    n = len(sorted_idx)
    if k_groups <= 0 or n < k_groups:
        return None
    gaps = []
    for i in range(n - 1):
        a = sorted_idx[i]; b = sorted_idx[i + 1]
        gaps.append((abs(values[b] - values[a]), i))
    gaps.sort(reverse=True, key=lambda t: t[0])
    cuts = sorted([pos for (_, pos) in gaps[:k_groups - 1]])
    groups = []
    start = 0
    for pos in cuts:
        groups.append(sorted_idx[start:pos + 1])
        start = pos + 1
    groups.append(sorted_idx[start:])
    return groups

def _row_groups_tightness(x, y, per_line):
    """
    Row-based grouping by y into n_rows bands; return (tightness, groups).
    Tightness is mean absolute deviation in y within bands (lower is tighter).
    """
    n = len(x)
    n_rows = n // per_line
    idx = list(range(n))
    idx.sort(key=lambda i: (y[i], x[i]))  # y asc, tie-break x
    bands = _largest_gaps_partition(idx, y, n_rows)
    if bands is None:
        return 1e30, None
    # average absolute deviation within each band
    dev = 0.0
    for b in bands:
        m = len(b)
        if m == 0:
            return 1e30, None
        mu = sum(y[i] for i in b) / m
        dev += sum(abs(y[i] - mu) for i in b) / m
    dev /= len(bands)
    return dev, bands

def _col_groups_tightness(x, y, per_line):
    """
    Column-based grouping by x into PER_LINE stacks; return (tightness, groups).
    Tightness is mean absolute deviation in x within stacks (lower is tighter).
    """
    n = len(x)
    idx = list(range(n))
    idx.sort(key=lambda i: (x[i], y[i]))  # x asc, tie-break y
    stacks = _largest_gaps_partition(idx, x, per_line)
    if stacks is None:
        return 1e30, None
    dev = 0.0
    for s in stacks:
        m = len(s)
        if m == 0:
            return 1e30, None
        mu = sum(x[i] for i in s) / m
        dev += sum(abs(x[i] - mu) for i in s) / m
    dev /= len(stacks)
    return dev, stacks

def _transpose_order(n_rows, n_cols):
    """
    Return permutation 'p' for transposing an n_rows x n_cols matrix
    stored in row-major order. After applying p, order is column-major.
    new_pos = (i % n_cols)*n_rows + (i // n_cols)
    """
    p = [0] * (n_rows * n_cols)
    for i in range(n_rows * n_cols):
        r = i // n_cols
        c = i %  n_cols
        j = c * n_rows + r
        p[i] = j
    return p

def _apply_perm_tokens(tokens, perm):
    out = [""] * len(tokens)
    for i, j in enumerate(perm):
        out[j] = tokens[i]
    return out

# ---------- main record logic (operates on floats, outputs original tokens) ----------

def process_record(rec):
    # Expect exactly: number;id;xs;ys;ws  (arrays comma-separated, no brackets)
    parts = rec.split(";", 4)
    if len(parts) < 5:
        return None
    number, sid, xs, ys, ws = (p.strip() for p in parts)

    x_tok, x = parse_tokens_and_floats(xs)
    y_tok, y = parse_tokens_and_floats(ys)
    w_tok, w = parse_tokens_and_floats(ws)

    n = len(x)
    if not (len(y) == n == len(w) and n > 0):
        return None

    if n % PER_LINE != 0:
        if STRICT_GRID:
            return None

    n_rows = n // PER_LINE

    # Decide Pattern A (rows) vs Pattern B (columns) from geometry
    row_dev, row_groups = _row_groups_tightness(x, y, PER_LINE)
    col_dev, col_groups = _col_groups_tightness(x, y, PER_LINE)

    is_pattern_b = col_dev < row_dev  # columns are tighter → Pattern B

    if not is_pattern_b:
        # Pattern A: ensure bottom→top in rows, left→right within rows
        if row_groups is None:
            idx = list(range(n))
            idx.sort(key=lambda i: (y[i], x[i]))  # fallback raster
            order = idx
        else:
            row_groups.sort(key=lambda g: sum(y[i] for i in g) / len(g))  # bottom → top
            order = []
            for g in row_groups:
                g.sort(key=lambda i: x[i])  # left → right within row
                order.extend(g)
        # Reorder TOKENS (preserve original formatting)
        x_tok = [x_tok[i] for i in order]
        y_tok = [y_tok[i] for i in order]
        w_tok = [w_tok[i] for i in order]
        return number, sid, x_tok, y_tok, w_tok

    # Pattern B: transpose from row-major to column-major using permutation
    perm = _transpose_order(n_rows, PER_LINE)
    x_tok = _apply_perm_tokens(x_tok, perm)
    y_tok = _apply_perm_tokens(y_tok, perm)
    w_tok = _apply_perm_tokens(w_tok, perm)
    return number, sid, x_tok, y_tok, w_tok

# ---------- IO ----------

def main():
    written = 0
    skipped = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        header = f_in.readline()
        has_header = header.lower().strip().startswith("number;")

        # Always write a header identical in structure
        out.write("number;id;nodes_x;nodes_y;weights\n")

        # If first line was data, process it
        if not has_header:
            line = header.strip()
            if line:
                item = process_record(line)
                if item is not None:
                    number, sid, x_tok, y_tok, w_tok = item
                    xs = ",".join(x_tok)
                    ys = ",".join(y_tok)
                    ws = ",".join(w_tok)
                    out.write(f"{number};{sid};{xs};{ys};{ws}\n")
                    written += 1
                else:
                    skipped += 1

        # Process remaining lines
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            item = process_record(line)
            if item is None:
                skipped += 1
                continue
            number, sid, x_tok, y_tok, w_tok = item
            xs = ",".join(x_tok)
            ys = ",".join(y_tok)
            ws = ",".join(w_tok)
            out.write(f"{number};{sid};{xs};{ys};{ws}\n")
            written += 1

    print(f"Wrote: {OUTPUT_FILE} | records: {written} | skipped: {skipped}")

if __name__ == "__main__":
    main()
