"""CAD export of the AeroSandbox airplane built in buildinguav.py.

Primary format : STEP (.step)  -> opens directly in Siemens NX (File > Import > STEP214/AP242).
Fallback format: STL  (.stl)   -> mesh only, imports into NX as a facet body (no editable features).

Requires for STEP:  pip install cadquery   (Python 3.9-3.12 is the safest choice)
"""
import os
import numpy as np
import aerosandbox as asb


def _scaled(airplane, scale):
    """Return a copy of the airplane scaled by `scale` (e.g. 1000 for m -> mm)."""
    if scale == 1:
        return airplane
    import copy
    ap = copy.deepcopy(airplane)
    for wing in ap.wings:
        for xs in wing.xsecs:
            xs.xyz_le = np.array(xs.xyz_le, dtype=float) * scale
            xs.chord = xs.chord * scale
    for fus in ap.fuselages:
        for xs in fus.xsecs:
            xs.xyz_c = np.array(xs.xyz_c, dtype=float) * scale
            xs.width = xs.width * scale
            xs.height = xs.height * scale
    return ap


def _write_stl(path, points, faces, name="uav"):
    """Write ASCII STL. Quads are split into two triangles."""
    tris = []
    for f in faces:
        f = list(f)
        if len(f) == 3:
            tris.append(f)
        else:  # quad
            tris.append([f[0], f[1], f[2]])
            tris.append([f[0], f[2], f[3]])
    with open(path, "w") as fh:
        fh.write(f"solid {name}\n")
        for a, b, c in tris:
            p0, p1, p2 = points[a], points[b], points[c]
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            n = n / norm if norm > 0 else n
            fh.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
            for p in (p0, p1, p2):
                fh.write(f"   vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write(f"endsolid {name}\n")


def export_stl(airplane, filename, scale=1000.0):
    """Export the whole airplane as one STL mesh (units = metres * scale)."""
    all_pts, all_faces, offset = [], [], 0
    for comp in list(airplane.wings) + list(airplane.fuselages):
        pts, faces = comp.mesh_body(method="quad")
        pts = np.asarray(pts, dtype=float) * scale
        faces = np.asarray(faces, dtype=int) + offset
        all_pts.append(pts)
        all_faces.extend(faces.tolist())
        offset += len(pts)
    _write_stl(filename, np.vstack(all_pts), all_faces, name=airplane.name or "uav")
    print(f"✓ STL written: {os.path.abspath(filename)}")
    return filename


def export_step(airplane, filename, scale=1000.0, min_te_thickness=0.001):
    """
    Export the whole airplane to one STEP file via AeroSandbox + CadQuery.

    scale            : unused (kept for API compatibility); output is always in mm.
    min_te_thickness : AeroSandbox thickens sharp trailing edges to this value (m,
                       before scaling) so the loft produces a valid solid.
    """
    # AeroSandbox's exporter ALREADY converts metres -> millimetres, so the airplane
    # must NOT be scaled again (doing so makes the part 1e6 times too large).
    # TE thickness is a fraction of chord, so it is not scaled either.
    airplane.export_cadquery_geometry(filename, minimum_airfoil_TE_thickness=min_te_thickness)
    print(f"✓ STEP written: {os.path.abspath(filename)}")
    return filename


def export_wing_step(wing, filename, scale=1000.0, min_te_thickness=0.002, n_spline_pts=None):
    """
    Build a wing solid directly in CadQuery (bypasses AeroSandbox's exporter).

    Each section: airfoil coordinates -> upper/lower splines + a flat trailing
    edge, scaled by chord, twisted about the leading edge (nose-up positive),
    placed at xyz_le. Neighbouring sections are joined with ruled lofts.
    Symmetric wings are mirrored about y = 0.
    """
    import cadquery as cq

    def section_wire(xsec):
        coords = np.array(xsec.airfoil.coordinates, dtype=float)
        i_le = int(np.argmin(coords[:, 0]))
        upper = coords[: i_le + 1][::-1].copy()   # LE -> TE
        lower = coords[i_le:].copy()              # LE -> TE
        # open the trailing edge by min_te_thickness (fraction of chord)
        upper[:, 1] += 0.5 * min_te_thickness * upper[:, 0]
        lower[:, 1] -= 0.5 * min_te_thickness * lower[:, 0]
        upper[0] = lower[0] = [0.0, 0.0]          # share the exact LE point

        chord = xsec.chord * scale
        th = np.radians(xsec.twist)
        le = np.array(xsec.xyz_le, dtype=float) * scale

        def to3d(p2):
            x, z = p2[0] * chord, p2[1] * chord
            xr = x * np.cos(th) + z * np.sin(th)       # nose-up twist about LE
            zr = -x * np.sin(th) + z * np.cos(th)
            return cq.Vector(le[0] + xr, le[1], le[2] + zr)

        up = [to3d(p) for p in upper]
        lo = [to3d(p) for p in lower][::-1]            # TE -> LE
        e_up = cq.Edge.makeSpline(up)
        e_te = cq.Edge.makeLine(up[-1], lo[0])
        e_lo = cq.Edge.makeSpline(lo)
        return cq.Wire.assembleEdges([e_up, e_te, e_lo])

    wires = [section_wire(xs) for xs in wing.xsecs]
    solids = [cq.Solid.makeLoft([wires[k], wires[k + 1]], True)
              for k in range(len(wires) - 1)]
    half = cq.Workplane("XY").add(solids[0])
    for sld in solids[1:]:
        half = half.union(cq.Workplane("XY").add(sld))
    if wing.symmetric:
        mirrored = half.mirror("XZ")
        half = half.union(mirrored)
    half = half.clean()
    body = half.val()
    n_solids = len(half.solids().vals())
    print(f"  {wing.name}: solids={n_solids}, valid={body.isValid()}, volume={body.Volume():.0f} mm^3")
    if n_solids != 1 or not body.isValid():
        print("  ⚠ wing did not fuse into one valid solid - exporting segments separately")
    cq.exporters.export(half, filename)
    print(f"✓ STEP written: {os.path.abspath(filename)}")
    return filename


def export_step_per_component(airplane, out_dir, scale=1000.0, min_te_thickness=0.001):
    """
    Robust fallback: one STEP file per wing / fin / fuselage. If a single
    component fails to loft (e.g. the zero-size fuselage nose section), the
    others are still exported. Assemble them in NX with the same origin.
    """
    os.makedirs(out_dir, exist_ok=True)
    ok, failed = [], []
    comps = [("wing", w) for w in airplane.wings] + [("fuselage", f) for f in airplane.fuselages]
    for kind, comp in comps:
        safe = comp.name.replace(" ", "_")
        path = os.path.join(out_dir, f"{kind}_{safe}.step")
        try:
            if kind == "wing" and any(
                    len(np.atleast_1d(xs.airfoil.coordinates)) > 0 for xs in comp.xsecs[:1]) \
                    and not comp.name.lower().startswith(("central fin", "left fin", "right fin")):
                export_wing_step(comp, path, scale=scale, min_te_thickness=max(min_te_thickness, 0.002))
            else:
                sub = asb.Airplane(
                    name=safe,
                    wings=[comp] if kind == "wing" else [],
                    fuselages=[comp] if kind == "fuselage" else [],
                )
                export_step(sub, path, scale=scale, min_te_thickness=min_te_thickness)
            ok.append(path)
        except Exception as exc:
            failed.append((comp.name, str(exc)))
            print(f"✗ {comp.name}: {exc}")
    return ok, failed


def export_airplane_cad(airplane, out_dir="output/cad", name="G1-T_UAV", scale=1000.0):
    """
    One call to get something NX can open.

    Order of attempts:
      1. single assembled STEP file
      2. per-component STEP files (if 1 fails)
      3. STL mesh (always written as a last-resort / visual reference)
    Returns a dict of the files produced.
    """
    os.makedirs(out_dir, exist_ok=True)
    result = {"step": None, "step_parts": [], "stl": None}

    step_path = os.path.join(out_dir, f"{name}.step")
    try:
        result["step"] = export_step(airplane, step_path, scale=scale)
    except Exception as exc:
        print(f"✗ Single-file STEP export failed: {exc}")
        print("  Trying per-component STEP export...")
        try:
            result["step_parts"], _ = export_step_per_component(
                airplane, os.path.join(out_dir, "parts"), scale=scale)
        except Exception as exc2:
            print(f"✗ Per-component export failed: {exc2}")

    try:
        result["stl"] = export_stl(airplane, os.path.join(out_dir, f"{name}.stl"), scale=scale)
    except Exception as exc:
        print(f"✗ STL export failed: {exc}")

    return result