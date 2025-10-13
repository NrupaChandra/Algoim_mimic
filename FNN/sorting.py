# sort_rows_or_transpose.py — keep row-major; transpose if column-major (no numpy/regex)

INPUT_FILE  = r"C:\Git\Algoim_mimic\FNN\Results\predicted_data_fnn.txt"
OUTPUT_FILE = r"C:\Git\Algoim_mimic\Weight_scalling\predicted_data_fnn_sorted.txt"

PER_LINE     = 8        # number of nodes per row (columns)
STRICT_GRID  = True     # require n % PER_LINE == 0 else skip record

# ---------- helpers ----------

def parse_arr(text):
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = s.split()
    return [float(p) for p in parts]

def fmt_block(vals, fmt, per_line=PER_LINE):
    out = ["["]
    for i in range(0, len(vals), per_line):
        chunk = vals[i:i+per_line]
        out.append(" " + "  ".join(fmt % v for v in chunk))
    out.append("]")
    return "\n".join(out)

def cut_one_record(buf):
    # Cut exactly one 'number;id;xs;ys;ws' record from buffer.
    pos = []
    start = 0
    for _ in range(4):
        k = buf.find(";", start)
        if k == -1:
            return None, buf
        pos.append(k)
        start = k + 1
    end_w = buf.find("]", pos[3] + 1)
    if end_w == -1:
        return None, buf
    return buf[:end_w + 1], buf[end_w + 1:]

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

def _apply_permutation(vals, perm):
    out = [0.0] * len(vals)
    for i, j in enumerate(perm):
        out[j] = vals[i]
    return out

# ---------- main record logic ----------

def process_record(rec):
    number, sid, xs, ys, ws = (p.strip() for p in rec.split(";", 4))
    x = parse_arr(xs); y = parse_arr(ys); w = parse_arr(ws)

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
        # Pattern A: leave as-is (already row-major raster), but ensure per-row left→right
        # Sort each detected row by x, and rows by their mean y (bottom→top)
        if row_groups is None:
            # conservative: fall back to simple y-then-x raster
            idx = list(range(n))
            idx.sort(key=lambda i: (y[i], x[i]))
            order = idx
        else:
            # bottom→top by mean y
            row_groups.sort(key=lambda g: sum(y[i] for i in g) / len(g))
            order = []
            for g in row_groups:
                g.sort(key=lambda i: x[i])  # left→right within row
                order.extend(g)
        x = [x[i] for i in order]
        y = [y[i] for i in order]
        w = [w[i] for i in order]
        return number, sid, x, y, w

    # Pattern B: apply your mapping — transpose row-major -> column-major indices.
    perm = _transpose_order(n_rows, PER_LINE)  # size n_rows * PER_LINE
    x = _apply_permutation(x, perm)
    y = _apply_permutation(y, perm)
    w = _apply_permutation(w, perm)
    return number, sid, x, y, w

# ---------- IO ----------

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        first = f.readline()
        has_header = first.lower().strip().startswith("number;")
        buf = "" if has_header else first
        buf += f.read()

    written = 0
    skipped = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("number;id;nodes_x;nodes_y;weights\n")
        while buf.strip():
            rec, buf = cut_one_record(buf)
            if rec is None:
                break
            item = process_record(rec)
            if item is None:
                skipped += 1
                continue
            number, sid, x, y, w = item
            xs = fmt_block(x, "%.8f")
            ys = fmt_block(y, "%.8f")
            ws = fmt_block(w, "%.8e")
            out.write(f"{number};{sid};{xs};{ys};{ws}\n")
            written += 1

    print(f"Wrote: {OUTPUT_FILE} | records: {written} | skipped: {skipped}")

if __name__ == "__main__":
    main()
