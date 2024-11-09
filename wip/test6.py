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

for botz in np.arange(11e-3, -13e-3, -2e-3):
    bounds = material.project().offset(1e-3, JoinType.Round)
    cut = Manifold.extrude(bounds-model.trim_by_plane([0,0,1], botz).project().offset(0.99e-3, JoinType.Round).offset(-0.99e-3, JoinType.Round), 2e-3).translate([0,0,botz])
    material -= cut

testcut = model.trim_by_plane([0,0,1], 0).trim_by_plane([0,0,-1], -1e-3)

ps.init()
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.register_surface_mesh("model", model.to_mesh().vert_properties[:,:3], model.to_mesh().tri_verts)
ps.register_surface_mesh("material", material.to_mesh().vert_properties[:,:3], material.to_mesh().tri_verts)
ps.register_surface_mesh("testcut", testcut.to_mesh().vert_properties[:,:3], testcut.to_mesh().tri_verts)
ps.show()