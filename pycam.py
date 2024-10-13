import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import time
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
import math
import sys
from manifold3d import Manifold, set_circular_segments
set_circular_segments(32)

mm.FixSelfIntersectionMethod
def loadobj(filename):
    model = mm.loadMesh(filename)
    model.transform(mm.AffineXf3f.translation(-model.getBoundingBox().center()))
    return model

def generate_bit(type, diameter, length=50e-3):
    if type == "endmill":
        return Manifold.cylinder(length, diameter/2)
    elif type == "ballnose":
        shaft = Manifold.cylinder(length-diameter/2, diameter/2)
        bit = Manifold.sphere(diameter/2)
        return shaft + bit
    elif type == "drill":
        conelen = 0.5*diameter/math.tan(math.radians(59))
        shaft = Manifold.cylinder(length-conelen, diameter/2)
        bit = Manifold.cylinder(conelen, diameter/2, 0).rotate(np.array([180, 0, 0]))
        return shaft + bit

class State:
    def __init__(self, model, material, tool, zero, loc):
        self.model = model
        self.material = material
        self.tool = tool
        self.zero = zero
        self.loc = loc

    def generate(filename, blank_size):
        model = loadobj(model_file)
        material = Manifold.cube(blank_size, True)
        tool = generate_bit("endmill", 4e-3)
        return State(model, material, tool, np.array([0,0,0]), np.array([0,0,0]))

    def init_polyscope(self, axis_scale=0.1):
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_front_dir("neg_y_front")
        ps.set_ground_plane_mode("none")
        ps.set_transparency_mode("pretty")
        ps.register_curve_network("linex", axis_scale*np.array([[0, 0, 0], [1, 0, 0]]), "line", color=[1, 0, 0], radius=0.002).set_transparency(0.5)
        ps.register_curve_network("liney", axis_scale*np.array([[0, 0, 0], [0, 1, 0]]), "line", color=[0, 1, 0], radius=0.002).set_transparency(0.5)
        ps.register_curve_network("linez", axis_scale*np.array([[0, 0, 0], [0, 0, 1]]), "line", color=[0, 0, 1], radius=0.002).set_transparency(0.5)
        for off in [0.125, 0.375, 0.625, 0.875]:
            ps.register_curve_network("linex"+str(off), -axis_scale*np.array([[off, 0, 0], [off+0.125, 0, 0]]), "line", color=[1, 0, 0], radius=0.002).set_transparency(0.5)
            ps.register_curve_network("liney"+str(off), -axis_scale*np.array([[0, off, 0], [0, off+0.125, 0]]), "line", color=[0, 1, 0], radius=0.002).set_transparency(0.5)
            ps.register_curve_network("linez"+str(off), -axis_scale*np.array([[0, 0, off], [0, 0, off+0.125]]), "line", color=[0, 0, 1], radius=0.002).set_transparency(0.5)
        ps.set_user_callback(polyscope_callback)
        ps.look_at([axis_scale, -axis_scale, axis_scale], [0, 0, 0], 1)

        ps.register_surface_mesh("model", mn.getNumpyVerts(self.model), mn.getNumpyFaces(self.model.topology))
        ps.register_surface_mesh("tool", self.tool.to_mesh().vert_properties[:, :3], self.tool.to_mesh().tri_verts).set_transparency(0.5)
        ps.register_surface_mesh("diff", self.material.to_mesh().vert_properties[:, :3], self.material.to_mesh().tri_verts).set_transparency(0.5)
        ps.show()

    def refresh_polyscape(self):
        ps.register_surface_mesh("model", mn.getNumpyVerts(self.model), mn.getNumpyFaces(self.model.topology))
        ps.register_surface_mesh("tool", self.tool.to_mesh().vert_properties[:, :3], self.tool.to_mesh().tri_verts).set_transparency(0.5)
        ps.register_surface_mesh("diff", self.material.to_mesh().vert_properties[:, :3], self.material.to_mesh().tri_verts).set_transparency(0.5)

class Operation:
    def __init__(self, state):
        self.state = state
        self.started = False

    def __iter__(self):
        self.prevState = State(mm.copyMesh(state.model), Manifold.compose([state.material]), Manifold.compose([state.tool]), [v for v in state.zero], [v for v in state.loc])
        self.started = True
        self.done = False
        return self

    def restore(self):
        self.state.model = self.prevState.model
        self.state.material = self.prevState.material
        self.state.tool = self.prevState.tool
        self.state.zero = self.prevState.zero
        self.state.loc = self.prevState.loc
        self.state.refresh_polyscape()

    def __next__(self):
        return self.state

class ToolChange(Operation):
    def __init__(self, state, tool, desc):
        self.tool = tool
        self.desc = desc
        super().__init__(state)

    def __next__(self):
        self.state.tool = self.tool
        self.state.loc = np.array([0, 0, 50e-3])
        self.state.tool = self.state.tool.translate(self.state.loc)
        state.refresh_polyscape()
        self.done = True
        raise StopIteration

    def __str__(self):
        return f"Change tool to " + self.desc

class RotateWorkpiece(Operation):
    def __init__(self, state, axis, angle, label):
        self.axis = axis
        self.angle = angle
        self.label = label
        super().__init__(state)

    def __next__(self):
        self.state.model.transform(mm.AffineXf3f.linear(mm.Matrix3f.rotation(self.axis, self.angle)))
        #self.state.material.transform(mm.AffineXf3f.linear(mm.Matrix3f.rotation(self.axis, self.angle)))
        state.refresh_polyscape()
        self.done = True
        raise StopIteration

    def __str__(self):
        return f"Rotate {math.degrees(self.angle):.0f} degrees around {self.label}-axis"	

class LinearInterpolate(Operation):
    def __init__(self, state, to, axesmask, n=30):
        self.to = to
        self.axesmask = axesmask
        if np.linalg.norm((state.loc-to)[:2],ord=2) < 1e-8:
            n = 1
        self.n = n
        super().__init__(state)

    def __iter__(self):
        self.i = 0
        self.delta = (self.to - self.state.loc)/(self.n) * self.axesmask
        return super().__iter__()

    def __next__(self):
        self.state.tool = self.state.tool.translate(self.delta)
        self.state.loc += self.delta
        self.state.material -= self.state.tool
        state.refresh_polyscape()
        self.i += 1
        if self.i < self.n:
            return self.state
        self.done = True
        raise StopIteration

    def __str__(self):
        return f"Linear interpolate to {self.to}, mask {self.axesmask}"

operations = []
operationid = 0
operationiter = None

tooldia = 4
rotangle = 90
lx, ly, lz = 0, 0, 50
cx, cy, cz = True, True, True

def polyscope_callback():
    global operations, operationid, operationiter, tooldia, rotangle, state, lx, ly, lz, cx, cy, cz

    play = False

    if (psim.TreeNode("Change Tool")):
        changed, tooldia = psim.InputFloat("Diameter [mm]", tooldia) 
        psim.TextUnformatted("Tool Type")
        psim.SameLine()
        if(psim.Button("End Mill")):
            operations.append(ToolChange(state, generate_bit("endmill", tooldia*1e-3), f"{tooldia:.1f} mm endmill"))
            play = True
        psim.SameLine()
        if(psim.Button("Ball Nose")):
            operations.append(ToolChange(state, generate_bit("ballnose", tooldia*1e-3), f"{tooldia:.1f} mm ballnose"))
            play = True
        psim.SameLine()
        if(psim.Button("Drill")):
            operations.append(ToolChange(state, generate_bit("drill", tooldia*1e-3), f"{tooldia:.1f} mm drill"))
            play = True
        psim.TreePop()
    
    if (psim.TreeNode("Rotate Workpiece")):
        changed, rotangle = psim.InputFloat("Angle [deg]", rotangle) 
        psim.TextUnformatted("Rotation Axis")
        psim.SameLine()
        if(psim.Button("X")):
            operations.append(RotateWorkpiece(state, mm.Vector3f(1, 0, 0), math.radians(rotangle), "X"))
            play = True
        psim.SameLine()
        if(psim.Button("Y")):
            operations.append(RotateWorkpiece(state, mm.Vector3f(0, 1, 0), math.radians(rotangle), "Y"))
            play = True
        psim.SameLine()
        if(psim.Button("Z")):
            operations.append(RotateWorkpiece(state, mm.Vector3f(0, 0, 1), math.radians(rotangle), "Z"))
            play = True
        psim.TreePop()
    
    if (psim.TreeNode("Linear Interpolate")):
        changed, cx = psim.Checkbox("X axis", cx) 
        if cx:
            psim.SameLine()
            changed, lx = psim.InputFloat("To [mm]", lx) 
        changed, cy = psim.Checkbox("Y axis", cy)
        if cy:
            psim.SameLine()
            changed, ly = psim.InputFloat("To [mm]2", ly)
        changed, cz = psim.Checkbox("Z axis", cz)
        if cz:
            psim.SameLine()
            changed, lz = psim.InputFloat("To [mm]3", lz)
        if(psim.Button("Add")):
            operations.append(LinearInterpolate(state, np.array([lx*1e-3, ly*1e-3, lz*1e-3]), np.array([1 if cx else 0, 1 if cy else 0, 1 if cz else 0])))
            play = True
        psim.TreePop()
    
    psim.TextUnformatted("Playback")
    psim.SameLine()
    if (psim.Button("< Back")):
        if operationiter is not None:
            operationiter = None
        else:
            operationid = max(0, operationid - 1)
        operations[operationid].restore()
    
    psim.SameLine()
    if (psim.Button("Delete")) and operationid < len(operations):
        operations[operationid].restore()
        operations.pop(operationid)

    psim.SameLine()
    if (play or psim.Button("Play >")) and operationid < len(operations):
        if operationiter is None:
            operationiter = iter(operations[operationid])
    
    if operationiter is not None:
        try:
            next(operationiter)
        except StopIteration:
            operationiter = None
            operationid += 1
        lx, ly, lz = state.loc*1e3
        
    psim.ListBox("Operations", operationid, [str(o) for o in operations])	
    

model_file = 'ledmain.obj'
blank_size = np.array([75e-3, 25e-3, 50e-3])
state = State.generate(model_file, blank_size)
state.loc = np.array([0, 0, 50e-3])
state.tool = state.tool.translate(state.loc)
state.init_polyscope()
