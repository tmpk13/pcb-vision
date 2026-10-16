#!/usr/bin/env python3
"""Dump a KiCad board's geometry through the pcbnew module as JSON.

The output is the ground truth the Rust parser's tests compare against:
footprint placements, absolute pad positions and sizes, drills, vias, tracks,
arcs, the board edge bounding box, and zone fills. Coordinates are in mm in
KiCad's board frame (x right, y down). Run with the system python that has
the pcbnew module, not the project's own environment.

    python3 tools/dump_board.py board.kicad_pcb > board.fixture.json

`--no-zones` leaves the zone fills out, for boards whose fills would make
the dump unwieldy.
"""
import json
import sys

import pcbnew


def mm(v):
    return round(pcbnew.ToMM(int(v)), 6)


def pt(v):
    return [mm(v.x), mm(v.y)]


def call_layered(obj, name, layer):
    """Call obj.name() with the per layer signature of newer KiCad if needed."""
    fn = getattr(obj, name)
    try:
        return fn()
    except TypeError:
        return fn(layer)


ATTR = {
    pcbnew.PAD_ATTRIB_PTH: "thru_hole",
    pcbnew.PAD_ATTRIB_SMD: "smd",
    pcbnew.PAD_ATTRIB_CONN: "connect",
    pcbnew.PAD_ATTRIB_NPTH: "np_thru_hole",
}

SHAPE = {
    pcbnew.PAD_SHAPE_CIRCLE: "circle",
    pcbnew.PAD_SHAPE_RECTANGLE: "rect",
    pcbnew.PAD_SHAPE_OVAL: "oval",
    pcbnew.PAD_SHAPE_TRAPEZOID: "trapezoid",
    pcbnew.PAD_SHAPE_ROUNDRECT: "roundrect",
    pcbnew.PAD_SHAPE_CHAMFERED_RECT: "chamfered_rect",
    pcbnew.PAD_SHAPE_CUSTOM: "custom",
}


def dump(path, zones=True):
    board = pcbnew.LoadBoard(path)
    out = {"footprints": [], "tracks": [], "arcs": [], "vias": [], "zones": [], "edges": []}
    for fp in board.GetFootprints():
        layer = board.GetLayerName(fp.GetLayer())
        pads = []
        for pad in fp.Pads():
            copper = pcbnew.F_Cu if layer == "F.Cu" else pcbnew.B_Cu
            size = call_layered(pad, "GetSize", copper)
            shape = call_layered(pad, "GetShape", copper)
            drill = pad.GetDrillSize()
            pads.append({
                "number": pad.GetNumber(),
                "kind": ATTR.get(pad.GetAttribute(), str(pad.GetAttribute())),
                "shape": SHAPE.get(shape, str(shape)),
                "at": pt(pad.GetPosition()),
                "rotation": round(pad.GetOrientationDegrees(), 4),
                "size": [mm(size.x), mm(size.y)],
                "drill": [mm(drill.x), mm(drill.y)],
                "layers": [board.GetLayerName(l) for l in pad.GetLayerSet().Seq()],
                "net": pad.GetNetname(),
            })
        out["footprints"].append({
            "reference": fp.GetReference(),
            "layer": layer,
            "at": pt(fp.GetPosition()),
            "rotation": round(fp.GetOrientationDegrees(), 4),
            "pads": pads,
        })
    for t in board.GetTracks():
        cls = t.GetClass()
        if cls == "PCB_VIA":
            out["vias"].append({
                "at": pt(t.GetPosition()),
                "size": mm(call_layered(t, "GetWidth", pcbnew.F_Cu)),
                "drill": mm(t.GetDrillValue()),
                "layers": [board.GetLayerName(l) for l in t.GetLayerSet().Seq()],
            })
        elif cls == "PCB_ARC":
            out["arcs"].append({
                "start": pt(t.GetStart()), "mid": pt(t.GetMid()), "end": pt(t.GetEnd()),
                "width": mm(t.GetWidth()), "layer": board.GetLayerName(t.GetLayer()),
            })
        else:
            out["tracks"].append({
                "start": pt(t.GetStart()), "end": pt(t.GetEnd()),
                "width": mm(t.GetWidth()), "layer": board.GetLayerName(t.GetLayer()),
            })
    bb = board.GetBoardEdgesBoundingBox()
    out["edge_bbox"] = [mm(bb.GetX()), mm(bb.GetY()), mm(bb.GetX() + bb.GetWidth()), mm(bb.GetY() + bb.GetHeight())]
    for d in board.GetDrawings():
        if d.GetClass() != "PCB_SHAPE" or board.GetLayerName(d.GetLayer()) != "Edge.Cuts":
            continue
        kind = d.ShowShape()
        rec = {"shape": kind, "start": pt(d.GetStart()), "end": pt(d.GetEnd())}
        if kind == "Arc":
            rec["mid"] = pt(d.GetArcMid())
        if kind == "Circle":
            rec["radius"] = mm(d.GetRadius())
        out["edges"].append(rec)
    for z in board.Zones() if zones else []:
        for layer in z.GetLayerSet().Seq():
            polys = z.GetFilledPolysList(layer)
            outlines = []
            for i in range(polys.OutlineCount()):
                chain = polys.Outline(i)
                outlines.append([pt(chain.CPoint(j)) for j in range(chain.PointCount())])
            if outlines:
                out["zones"].append({"layer": board.GetLayerName(layer), "net": z.GetNetname(), "outlines": outlines})
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    json.dump(dump(args[0], zones="--no-zones" not in sys.argv), sys.stdout, indent=1)
