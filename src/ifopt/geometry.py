"""Optical layout taken from ``system.xml`` (draw.io); used by the combiner configs and their tests.

The filter plate is the thick line (364,480)-(464,580): a plate at 45 deg.  Screen y grows
downwards.  The two sources illuminate OPPOSITE faces of the plate:

  source 1 (red,  "wavelength 1"): travels right->left, hits the upper-right face
      T1 continues left, R1 is reflected up
  source 2 (green,"wavelength 2"): travels bottom->top, hits the lower-left face
      T2 continues up,   R2 is reflected left

Output ports: left = {T1, R2}, up = {R1, T2}.  The plate is a dichroic beam combiner, so
"maximise T1 and R2" means "send both beams into the left port".

Both beams lie in the drawing plane, which is therefore the plane of incidence of both:
s-polarisation = E-field perpendicular to the drawing.

``verify_against_drawio`` re-derives this table from the XML (arrow colours, directions and
nearest labels) so the hard-coded mapping is checked against the drawing in the tests.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_FACE = {1: "upper-right face", 2: "lower-left face"}
OUTPUT_PORT = {"T1": "left", "R1": "up", "T2": "up", "R2": "left"}


def expected_port_terms(path: str | Path) -> dict[str, set[str]]:
    """From the drawing: port -> report terms in target-table notation, e.g. 'left' -> {'T@1', 'R@2'}
    (beam labels '1' and '2' = source numbers)."""
    out: dict[str, set[str]] = {}
    for coef, port in verify_against_drawio(path).items():
        out.setdefault(port, set()).add(f"{coef[0]}@{coef[1]}")
    return out


def verify_against_drawio(path: str | Path) -> dict[str, str]:
    """Derive {label: port} from the drawing; raises AssertionError if inconsistent."""
    root = ET.parse(path).getroot()
    edges, labels, filt = [], {}, None
    for cell in root.iter("mxCell"):
        style = cell.get("style", "")
        geo = cell.find("mxGeometry")
        if geo is None:
            continue
        if cell.get("edge") == "1":
            pts = {p.get("as"): (float(p.get("x")), float(p.get("y"))) for p in geo.findall("mxPoint")}
            if "sourcePoint" not in pts or "targetPoint" not in pts:
                continue
            color = "red" if "B20000" in style else "green" if "2D7600" in style else "other"
            if "strokeWidth=4" in style:
                filt = (pts["sourcePoint"], pts["targetPoint"])
            elif "endArrow=classic" in style:
                edges.append((color, pts["sourcePoint"], pts["targetPoint"]))
        elif cell.get("vertex") == "1":
            x, y = float(geo.get("x")), float(geo.get("y"))
            w, h = float(geo.get("width")), float(geo.get("height"))
            labels[cell.get("value")] = (x + w / 2, y + h / 2)
    assert filt is not None, "filter line not found"
    (fx0, fy0), (fx1, fy1) = filt
    assert abs(abs(math.degrees(math.atan2(fy1 - fy0, fx1 - fx0))) - 45) < 1e-6, "filter is not at 45 deg"

    def dist_to_filter(p):
        nx, ny = fy1 - fy0, -(fx1 - fx0)
        return abs((p[0] - fx0) * nx + (p[1] - fy0) * ny) / math.hypot(nx, ny)

    def direction(a, b):
        d = (b[0] - a[0], b[1] - a[1])
        n = math.hypot(*d)
        return (round(d[0] / n), round(d[1] / n))

    incoming = {c: direction(a, b) for c, a, b in edges if dist_to_filter(b) < dist_to_filter(a)}
    source_of = {"red": 1, "green": 2}
    result = {}
    for color, a, b in edges:
        if dist_to_filter(b) < dist_to_filter(a):
            continue  # incident beam
        kind = "T" if direction(a, b) == incoming[color] else "R"
        name = f"{kind}{source_of[color]}"
        nearest = min(("T1", "T2", "R1", "R2"), key=lambda lab: math.dist(labels[lab], b))
        assert nearest == name, f"arrow {color} {kind} ends nearest to label {nearest}, expected {name}"
        d = direction(a, b)
        result[name] = {(-1, 0): "left", (0, -1): "up", (1, 0): "right", (0, 1): "down"}[d]
    return result
