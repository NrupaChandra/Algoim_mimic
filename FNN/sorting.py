# sorting_rows.py — strict raster order (rows by y, left→right by x), no numpy/regex

INPUT_FILE  = r"C:\Git\Algoim_mimic\FNN\Results\predicted_data_fnn.txt"
OUTPUT_FILE = r"C:\Git\Algoim_mimic\Weight_scalling\predicted_data_fnn_sorted.txt"

PER_LINE     = 8       # number of nodes per row (8x8 grid => 8)
STRICT_GRID  = True    # if True, require n % PER_LINE == 0 else skip record

def parse_arr(text):
    """Parse a bracketed, whitespace-separated array into list[float]."""
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = s.split()
    return [float(p) for p in parts]

def fmt_block(vals, fmt, per_line=PER_LINE):
    """Format values in a bracketed, multi-line block with per_line numbers."""
    out = ["["]
    for i in range(0, len(vals), per_line):
        chunk = vals[i:i+per_line]
        line = " " + "  ".join(fmt % v for v in chunk)
        out.append(line)
    out.append("]")
    return "\n".join(out)

def cut_one_record(buf):
    """
    Cut exactly one 'number;id;xs;ys;ws' record from buffer.
    Looks for four semicolons and the closing ']' of the final array.
    """
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

def process_record(rec):
    """
    Enforce raster scan:
      - Sort all points by y (ascending) to stack rows bottom→top.
      - Partition into consecutive blocks of PER_LINE to form rows.
      - Sort each row by x (ascending) to ensure left→right.
    Falls back or skips based on STRICT_GRID if count isn't a clean multiple of PER_LINE.
    """
    number, sid, xs, ys, ws = (p.strip() for p in rec.split(";", 4))
    x = parse_arr(xs); y = parse_arr(ys); w = parse_arr(ws)

    n = len(x)
    if not (len(y) == n == len(w) and n > 0):
        return None

    if n % PER_LINE != 0:
        if STRICT_GRID:
            # Not a clean grid; skip this record
            return None
        else:
            # Fallback: stable sort by (y, x) and then wrap every PER_LINE
            order = sorted(range(n), key=lambda i: (y[i], x[i]))
    else:
        rows = n // PER_LINE
        # 1) sort globally by y then x to get bottom→top, and stable left bias
        by_y = sorted(range(n), key=lambda i: (y[i], x[i]))
        # 2) slice consecutive blocks of PER_LINE as rows
        # 3) sort each row by x only (final left→right guarantee)
        order = []
        for r in range(rows):
            block = by_y[r*PER_LINE:(r+1)*PER_LINE]
            block_sorted = sorted(block, key=lambda i: x[i])
            order.extend(block_sorted)

    x = [x[i] for i in order]
    y = [y[i] for i in order]
    w = [w[i] for i in order]
    return number, sid, x, y, w

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
