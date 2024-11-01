import numpy as np
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
import polyscope as ps
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pyclipr
from manifold3d import Manifold, set_circular_segments, CrossSection, Mesh, FillRule, JoinType
set_circular_segments(32)


def loadobj(filename):
    model = mm.loadMesh(filename)
    model.transform(mm.AffineXf3f.translation(-model.getBoundingBox().center()))
    mesh = Mesh(vert_properties = mn.getNumpyVerts(model), tri_verts = mn.getNumpyFaces(model.topology))   
    return Manifold(mesh)

model_file = 'ledmain.obj'
blank_size = np.array([75e-3, 25e-3, 50e-3])
model = loadobj(model_file).rotate([90,0,0])
material = Manifold.cube(blank_size, True).rotate([90,0,0])


endz = 0e-3
stepz = 4e-3
stock = 0.1e-3


cut = model.trim_by_plane([0,0,1], endz+stock)
verts = cut.to_mesh().vert_properties[:,:3]
pts = []
for tri in cut.to_mesh().tri_verts:
    zs = verts[tri,2]
    if max(zs)-min(zs) < 1e-9:
        x1, y1, x2, y2, x3, y3 = *verts[tri[0],:2], *verts[tri[1],:2], *verts[tri[2],:2]
        area = -0.5*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        if abs(area) > 1e-12:
            pts.append([zs[0], area])
pts.sort(key=lambda x: x[1])
zstops = []
for pt in pts:
    if not any(abs(zstop-pt[0]) < 1e-9 for zstop in zstops):
        zstops.append(pt[0])
zstops = sorted(zstops)[::-1]
startz = cut.bounding_box()[5]
nslices = math.ceil((startz-endz-stock)/stepz)
zsi = 0
fzstops = []
for pz in [endz + i*stepz + stock for i in range(nslices)][::-1]:
    fzstops.append(pz)
    while zsi < len(zstops) and zstops[zsi]+stock > pz-1e-9:
        if zstops[zsi]+stock > pz+1e-9:
            fzstops.append(zstops[zsi]+stock)
        zsi += 1

print(fzstops)