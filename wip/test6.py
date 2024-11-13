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
model = loadobj(model_file).rotate([-90,0,0])
material = Manifold.cube(blank_size, True).rotate([90,0,0])

#for botz in np.arange(11e-3, -13e-3, -2e-3):
#    bounds = material.project().offset(1e-3, JoinType.Round)
#    cut = Manifold.extrude(bounds-model.trim_by_plane([0,0,1], botz).project().offset(0.99e-3, JoinType.Round).offset(-0.99e-3, JoinType.Round), 2e-3).translate([0,0,botz])
#    material -= cut

top = 10.5e-3
step = 1e-3
testcut = model.trim_by_plane([0,0,1], top-step).trim_by_plane([0,0,-1], -top)

#ps.init()
#ps.set_up_dir("z_up")
#s.set_front_dir("neg_y_front")
#ps.register_surface_mesh("model", model.to_mesh().vert_properties[:,:3], model.to_mesh().tri_verts)
#ps.register_surface_mesh("material", material.to_mesh().vert_properties[:,:3], material.to_mesh().tri_verts)
#ps.register_surface_mesh("testcut", testcut.to_mesh().vert_properties[:,:3], testcut.to_mesh().tri_verts)
#ps.show()

verts, faces = testcut.to_mesh().vert_properties[:,:3], testcut.to_mesh().tri_verts
facenorms = [np.cross(verts[face[1]]-verts[face[0]], verts[face[2]]-verts[face[0]]) for face in faces]
facenorms = [norm/np.linalg.norm(norm) for norm in facenorms]
topslopers = [faceidx for faceidx, norm in enumerate(facenorms) if norm[2] > 1e-2 and np.linalg.norm(norm[:2]) > 1e-2 and sum(1 for vi in faces[faceidx] if verts[vi,2] >= top-1e-8) == 2]
bitpts = [sum(verts[vi] for vi in faces[fi])/3 + facenorms[fi]*1.5e-3 for fi in topslopers]
chains = []
chainpos = {}
for faceidx in topslopers:
    topverts = [tuple([int(v*1e6) for v in verts[vi]]) for vi in faces[faceidx] if verts[vi,2] >= top-1e-8]
    if topverts[0] in chainpos:
        chainidx, inidx = chainpos[topverts[0]]
        chains[chainidx].insert(inidx, [topverts[1], faceidx])
        if inidx == -1:
            chains[chainidx][-2][1] = faceidx
        chainpos[topverts[1]] = (chainidx, inidx)
    elif topverts[1] in chainpos:
        chainidx, inidx = chainpos[topverts[1]]
        chains[chainidx].insert(inidx, [topverts[0], faceidx])
        if inidx == -1:
            chains[chainidx][-2][1] = faceidx
        chainpos[topverts[0]] = (chainidx, inidx)
    else:
        chainpos[topverts[0]] = (len(chains), 0)
        chainpos[topverts[1]] = (len(chains), -1)
        chains.append([(topverts[0], faceidx), (topverts[1], None)])

plt.figure()
plt.axis('equal')
for faceidx in topslopers:
    face = faces[faceidx]
    plt.gca().add_patch(Polygon(verts[face,:2], edgecolor='black', facecolor='none'))
for path in chains:
    pts = np.array([np.array([float(v)/1e6 for v in pt[0]]) for pt in path])
    plt.plot(pts[:,0], pts[:,1], 'r-+', markeredgewidth=1)
plt.show()

print([pt[2] for pt in bitpts])