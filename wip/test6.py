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

#for botz in np.arange(11e-3, -13e-3, -2e-3):
#    bounds = material.project().offset(1e-3, JoinType.Round)
#    cut = Manifold.extrude(bounds-model.trim_by_plane([0,0,1], botz).project().offset(0.99e-3, JoinType.Round).offset(-0.99e-3, JoinType.Round), 2e-3).translate([0,0,botz])
#    material -= cut

top = 2.1e-3
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
chaincnt = 0
chains = {}
chainendidxs = {}
chainstartidxs = {}
for faceidx in topslopers:
    faceswrap = np.concatenate([faces[faceidx], faces[faceidx]])
    faceswrap = list(zip(faceswrap[:-1], faceswrap[1:]))
    topverts = [tuple([int(v*1e6) for v in verts[vi]]) for vi in [pair for pair in faceswrap if verts[pair[0],2] >= top-1e-8 and verts[pair[1],2] >= top-1e-8][0]]
    idx1 = chainendidxs[topverts[0]] if topverts[0] in chainendidxs else None
    idx2 = chainstartidxs[topverts[1]] if topverts[1] in chainstartidxs else None
    if (idx1 is not None) and (idx2 is not None) and (idx1 != idx2):
        idx2 = chainstartidxs[topverts[1]]
        chains[idx2][0] = (topverts[1], faceidx)
        chains[idx1] += chains[idx2]
        chains.pop(idx2)
        chainendidxs[chains[idx1][-1][0]] = idx1
        chainendidxs.pop(topverts[0])
        chainstartidxs.pop(topverts[1])
    elif idx1 is not None:
        chains[idx1].append((topverts[1], faceidx))
        chainendidxs[topverts[1]] = idx1
        chainendidxs.pop(topverts[0])
    elif idx2 is not None:
        chains[idx2].insert(0, (topverts[0], None))
        chains[idx2][1] = (topverts[1], faceidx)
        chainstartidxs[topverts[0]] = idx2
        chainstartidxs.pop(topverts[1])
    else:
        chains[chaincnt] = [(topverts[0], None), (topverts[1], faceidx)]
        chainendidxs[topverts[1]] = chaincnt
        chainstartidxs[topverts[0]] = chaincnt
        chaincnt += 1
paths = []
for pathi in chains:
    path = chains[pathi]
    pts = [pt[0] for pt in path]
    fcs = [[path[1][1]]] + [[path[i][1], path[i+1][1]] for i in range(1,len(path)-1)] + [[path[-1][1]]]
    if pts[0] == pts[-1]:
        fcs[0].append(path[-1][1])
        fcs[-1].append(path[1][1])
    newpath = [np.array([float(v)/1e6 for v in pts[i]]) + 1.5e-3*np.average(np.array([facenorms[fi] for fi in fcs[i]]), axis=0) for i in range(len(path))]
    paths.append(newpath)


plt.figure()
plt.axis('equal')
#for faceidx in topslopers:
    #face = faces[faceidx]
for face in faces:
    plt.gca().add_patch(Polygon(verts[face,:2], edgecolor='black', facecolor='none'))
for path1,path2 in zip(chains,paths):
    pts1 = np.array([np.array([float(v)/1e6 for v in pt[0]]) for pt in chains[path1]])
    pts2 = np.array(path2)
    plt.plot(pts2[:,0], pts2[:,1], 'r-+', markeredgewidth=1)
    for pair in zip(pts1, pts2):
        plt.plot([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]], 'g-')
plt.show()

print([pt[2] for pt in bitpts])