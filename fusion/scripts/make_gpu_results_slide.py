#!/usr/bin/env /usr/bin/python3
"""Create a one-slide, editable PowerPoint from saved SpMM-SpMM results.

Uses LibreOffice's UNO API, available with the system Python on this machine.
The slide uses fixed tile size 8 for tile-based variants; it never selects the
best tile from the results. Only CSR-CSC atomic is tuned, selecting among
1, 2, 4, and 8 rows per block for each matrix/order case.
"""

import argparse
import csv
import math
import socket
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import uno
from com.sun.star.awt import Point, Size
from com.sun.star.beans import PropertyValue


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHODS = (
    ("cuSPARSE ALG3", "GPU_Unfused_CuSparse_ALG3"),
    ("Row-wise", "GPU_Unfused_SeqReduceRowBalance"),
    ("Row-wise · tile 8", "GPU_Unfused_SeqReduceRowCoarsened_8"),
    ("CSR–CSR fused · tile 8", "GPU_Fused_Reordered_HighFusionRatio_8"),
    ("CSR–CSR atomic · tile 8", "GPU_AtomicFused_Reordered_HighFusionRatio_8"),
    ("CSR–CSC atomic · tuned", "GPU_CSRCSC_Fused_Atomic"),
)


def prop(name, value):
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def rgb(code):
    return int(code.lstrip("#"), 16)


def read_results(cohort):
    path = RESULTS / f"gpu_all_impl_{cohort}_16t_32f_summary.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["implementation"]].append(row)
    output = {}
    for _, key in METHODS:
        selected = grouped[key]
        ratios = [float(row["speedup_vs_row_balance"]) for row in selected]
        output[key] = {
            "speedup": math.exp(statistics.mean(map(math.log, ratios))),
            "correct": sum(int(row["correct_trials"]) for row in selected),
            "trials": sum(int(row["trials"]) for row in selected),
        }
    tuned_path = RESULTS / "gpu_csrcsc_atomic_tuned_16t_32f.csv"
    with tuned_path.open(newline="") as stream:
        tuned = [row for row in csv.DictReader(stream)
                 if row["cohort"] == cohort]
    if len(tuned) != (18 if cohort == "end2end" else 8):
        raise RuntimeError(f"Incomplete tuned CSR-CSC results for {cohort}")
    output["GPU_CSRCSC_Fused_Atomic"] = {
        "speedup": math.exp(statistics.mean(
            math.log(float(row["speedup_vs_row_balance"]))
            for row in tuned)),
        "correct": sum(int(row["correct_trials"]) for row in tuned),
        "trials": sum(int(row["trials"]) for row in tuned),
    }
    return output


def connect_office():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    profile = tempfile.TemporaryDirectory(prefix="spmm_slide_lo_")
    process = subprocess.Popen(
        ["libreoffice", "-env:UserInstallation=" +
         uno.systemPathToFileUrl(profile.name), "--headless", "--nodefault",
         "--norestore", "--nofirststartwizard",
         f"--accept=socket,host=127.0.0.1,port={port};urp;StarOffice.ServiceManager"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(100):
        try:
            remote = resolver.resolve(
                f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext")
            manager = remote.ServiceManager
            desktop = manager.createInstanceWithContext(
                "com.sun.star.frame.Desktop", remote)
            return desktop, process, profile
        except Exception:
            if process.poll() is not None:
                break
            time.sleep(0.1)
    process.terminate()
    profile.cleanup()
    raise RuntimeError("Could not connect to headless LibreOffice")


def rectangle(doc, page, x, y, w, h, fill):
    shape = doc.createInstance("com.sun.star.drawing.RectangleShape")
    shape.Position = Point(int(x), int(y))
    shape.Size = Size(int(w), int(h))
    shape.FillStyle = uno.Enum("com.sun.star.drawing.FillStyle", "SOLID")
    shape.FillColor = rgb(fill)
    shape.LineStyle = uno.Enum("com.sun.star.drawing.LineStyle", "NONE")
    page.add(shape)
    return shape


def label(doc, page, x, y, w, h, value, size, color, bold=False,
          center=False):
    shape = doc.createInstance("com.sun.star.drawing.TextShape")
    shape.Position = Point(int(x), int(y))
    shape.Size = Size(int(w), int(h))
    page.add(shape)
    shape.String = value
    shape.FillStyle = uno.Enum("com.sun.star.drawing.FillStyle", "NONE")
    shape.LineStyle = uno.Enum("com.sun.star.drawing.LineStyle", "NONE")
    shape.TextLeftDistance = 0
    shape.TextRightDistance = 0
    shape.TextUpperDistance = 0
    shape.TextLowerDistance = 0
    shape.TextVerticalAdjust = uno.Enum(
        "com.sun.star.drawing.TextVerticalAdjust", "CENTER")
    cursor = shape.createTextCursor()
    cursor.CharFontName = "Liberation Sans"
    cursor.CharHeight = float(size)
    cursor.CharWeight = 150.0 if bold else 100.0
    cursor.CharColor = rgb(color)
    if center:
        cursor.ParaAdjust = uno.Enum(
            "com.sun.star.style.ParagraphAdjust", "CENTER")
    return shape


def make_slide(doc, end2end, spd):
    page = doc.getDrawPages().getByIndex(0)
    page.Width = 33867
    page.Height = 19050
    W, H = page.Width, page.Height
    dark = "#101B2B"
    muted = "#9DACC0"
    white = "#F5F8FC"
    cyan = "#66D6D0"
    green = "#78D8A3"
    amber = "#F4BE72"

    rectangle(doc, page, 0, 0, W, H, dark)
    rectangle(doc, page, 1450, 1200, 150, 1160, cyan)
    label(doc, page, 1850, 1100, 29000, 1200,
          "GPU SpMM–SpMM  |  Implementation comparison", 28, white, True)
    label(doc, page, 1850, 2440, 29000, 560,
          "Geometric-mean speedup over row-wise unfused GPU  ·  higher is better",
          14, muted)

    left, top, width = 1450, 3750, 30967
    col = (13700, 5600, 5600, 6067)
    header_h, row_h = 900, 1480
    rectangle(doc, page, left, top, width, header_h, "#263A50")
    labels = ("METHOD", "END-TO-END", "SPD", "VERIFICATION")
    at = left
    for i, (title, cw) in enumerate(zip(labels, col)):
        label(doc, page, at + (360 if i == 0 else 0), top + 75,
              cw - (360 if i == 0 else 0), header_h - 150,
              title, 11, muted, True, center=i != 0)
        at += cw

    for i, (name, key) in enumerate(METHODS):
        y = top + header_h + i * row_h
        rectangle(doc, page, left, y, width, row_h - 18,
                  "#1D2C3F" if i % 2 == 0 else "#223347")
        if key == "GPU_CSRCSC_Fused_Atomic":
            rectangle(doc, page, left, y, 130, row_h - 18, cyan)
        e, s = end2end[key], spd[key]
        ok = (e["correct"] == e["trials"] and
              s["correct"] == s["trials"])
        validation = "All passed" if ok else \
            f"SPD {s['correct']}/{s['trials']}"
        label(doc, page, left + 360, y + 150, col[0] - 600,
              row_h - 300, name, 16, white,
              bold=key == "GPU_CSRCSC_Fused_Atomic")
        label(doc, page, left + col[0], y + 125, col[1], row_h - 250,
              f"{e['speedup']:.2f}×", 20,
              green if e["speedup"] > 1.01 else
              (amber if e["speedup"] < 0.95 else white), True, True)
        label(doc, page, left + col[0] + col[1], y + 125,
              col[2], row_h - 250, f"{s['speedup']:.2f}×", 20,
              green if s["speedup"] > 1.01 else
              (amber if s["speedup"] < 0.95 else white), True, True)
        label(doc, page, left + sum(col[:3]), y + 125, col[3], row_h - 250,
              validation, 13, green if ok else amber, True, True)

    callout_y = top + header_h + len(METHODS) * row_h + 350
    rectangle(doc, page, left, callout_y, width, 1000, "#153C43")
    label(doc, page, left + 360, callout_y + 80, width - 720, 830,
          "Takeaway  ·  tuning narrows the CSR–CSC gap, but cuSPARSE ALG3 still leads.",
          16, "#C5F7E0", True)

    label(doc, page, left, H - 2250, width, 450,
          "RTX 3060  ·  32 features  ·  9 end-to-end + 4 SPD matrices, original & RCM  ·  3 repeats × 7 trials; mean 1–6",
          10, muted)
    label(doc, page, left, H - 1720, width, 450,
          "Tile variants fixed at tile 8. Only CSR–CSC atomic tuned: best mean of 1, 2, 4, 8 rows/block per case.",
          10, muted)
    label(doc, page, left, H - 1190, width, 450,
          "Verification uses a strict 10⁻⁶ absolute-error threshold; failed SPD trials are flagged, not hidden.",
          10, muted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="output .pptx file")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pptx":
        parser.error("output must end in .pptx")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    desktop, process, profile = connect_office()
    doc = None
    try:
        doc = desktop.loadComponentFromURL(
            "private:factory/simpress", "_blank", 0, (prop("Hidden", True),))
        make_slide(doc, read_results("end2end"), read_results("spd"))
        doc.storeAsURL(uno.systemPathToFileUrl(str(args.output.resolve())),
                       (prop("FilterName", "Impress MS PowerPoint 2007 XML"),
                        prop("Overwrite", True)))
        pdf = args.output.with_suffix(".pdf")
        doc.storeToURL(uno.systemPathToFileUrl(str(pdf.resolve())),
                       (prop("FilterName", "impress_pdf_Export"),
                        prop("Overwrite", True)))
        print(args.output)
        print(pdf)
    finally:
        if doc is not None:
            doc.close(True)
        process.terminate()
        process.wait(timeout=10)
        profile.cleanup()


if __name__ == "__main__":
    main()
