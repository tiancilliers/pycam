import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import time
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
import math
import sys
from copy import deepcopy
from manifold3d import Manifold, set_circular_segments, CrossSection, Mesh
set_circular_segments(32)

Manifold.__deepcopy__ = lambda self, memo: Manifold.compose([self])

def loadobj(filename):
    model = mm.loadMesh(filename)
    model.transform(mm.AffineXf3f.translation(-model.getBoundingBox().center()))
    mesh = Mesh(vert_properties = mn.getNumpyVerts(model), tri_verts = mn.getNumpyFaces(model.topology))   
    return Manifold(mesh)


class Tool:
    def __init__(self, type, diameter, length=50e-3):
        self.type = type
        self.diameter = diameter
        self.length = length
    
    def generate_bit(self, loc):
        if self.type == "endmill":
            return Manifold.cylinder(self.length, self.diameter/2).translate(loc)
        elif self.type == "ballnose":
            shaft = Manifold.cylinder(self.length-self.diameter/2, self.diameter/2)
            bit = Manifold.sphere(self.diameter/2)
            return (shaft + bit).translate([0,0,self.diameter/2]).translate(loc)
        elif self.type == "drill":
            conelen = 0.5*self.diameter/math.tan(math.radians(59))
            shaft = Manifold.cylinder(self.length-conelen, self.diameter/2)
            bit = Manifold.cylinder(conelen, self.diameter/2, 0).rotate(np.array([180, 0, 0])).translate(conelen)
            return (shaft + bit).translate(loc)
        
    def generate_linear_cutarea(self, loc1, loc2):
        delta = loc2 - loc1
        dcyl = 0 if self.type == "endmill" else (0.5*self.diameter if self.type == "ballnose" else 0.5*self.diameter/math.tan(math.radians(59)))
        
        dxy = (delta[0]**2 + delta[1]**2)**0.5
        dxyz = (delta[0]**2 + delta[1]**2 + delta[2]**2)**0.5
        phi = math.atan2(delta[0], delta[1])+np.pi
        theta = math.atan2(dxy, delta[2])

        cutarea = Manifold()
        if dxy > 1e-9:
            rectrod = CrossSection.square([self.diameter, (self.length-dcyl)*np.sin(theta)]).translate([-0.5*self.diameter, 0])
            cutarea += Manifold.extrude(rectrod, dxyz)\
                .warp(lambda v: [v[0], v[1], v[2]+v[1]/math.tan(theta)])\
                .rotate(np.array([np.rad2deg(theta), 0, 0]))\
                .translate([0,0,dcyl])
        if abs(delta[2]) > 1e-9:
            circ = CrossSection.circle(0.5*self.diameter)
            cutarea += Manifold.extrude(circ, 1).scale([1,1,delta[2]])\
                .warp(lambda v: [v[0], v[1]-v[2]*math.tan(theta), v[2]])\
                .translate([0,0,dcyl])
        cutarea += Manifold.cylinder(self.length-dcyl, 0.5*self.diameter).rotate([0, 0, np.degrees(phi)]).translate([0,0,dcyl])
        cutarea += Manifold.cylinder(self.length-dcyl, 0.5*self.diameter).rotate([0, 0, np.degrees(phi)]).translate([0,-dxy,delta[2]+dcyl])
        if self.type == "ballnose":
            circ = CrossSection.circle(0.5*self.diameter)
            cutarea += Manifold.extrude(circ, dxyz)\
                .rotate(np.array([np.rad2deg(theta), 0, 0]))\
                .translate([0,0,dcyl])
            if dxy > 1e-9:
                cutarea += Manifold.revolve(circ, circular_segments=int(16*(np.pi-theta)/np.pi)+1, revolve_degrees=np.degrees(np.pi-theta))\
                    .rotate(np.array([-90, 0., 90]))\
                    .translate([0,0,dcyl])
                cutarea += Manifold.revolve(circ, circular_segments=int(16*(theta)/np.pi)+1, revolve_degrees=np.degrees(theta))\
                    .rotate(np.array([-90, 0., -90]))\
                    .translate([0,-dxy,dcyl+delta[2]])
            else:
                cutarea += Manifold.sphere(0.5*self.diameter).translate([0,0,dcyl])

        cutarea = cutarea.rotate(np.array([0, 0, -np.rad2deg(phi)])).translate(loc1)

        return cutarea

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
        tool = Tool("endmill", 4e-3)
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

        ps.register_surface_mesh("model", self.model.to_mesh().vert_properties[:, :3], self.model.to_mesh().tri_verts)
        ps.register_surface_mesh("tool", self.tool.generate_bit(self.loc).to_mesh().vert_properties[:, :3], self.tool.generate_bit(self.loc).to_mesh().tri_verts).set_transparency(0.5)
        ps.register_surface_mesh("diff", self.material.to_mesh().vert_properties[:, :3], self.material.to_mesh().tri_verts).set_transparency(0.5)
        ps.show()

    def refresh_polyscape(self):
        ps.register_surface_mesh("model", self.model.to_mesh().vert_properties[:, :3], self.model.to_mesh().tri_verts)
        ps.register_surface_mesh("tool", self.tool.generate_bit(self.loc).to_mesh().vert_properties[:, :3], self.tool.generate_bit(self.loc).to_mesh().tri_verts).set_transparency(0.5)
        ps.register_surface_mesh("diff", self.material.to_mesh().vert_properties[:, :3], self.material.to_mesh().tri_verts).set_transparency(0.5)

class Operation:
    def __init__(self, state):
        self.state = state
        self.started = False

    def __iter__(self):
        self.prevState = deepcopy(self.state)
        self.started = True
        self.done = False
        return self

    def restore(self):
        self.state = deepcopy(self.prevState)
        self.state.refresh_polyscape()
        self.started = True
        self.done = False
        return self.state

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
        state.refresh_polyscape()
        self.done = True
        raise StopIteration

    def __str__(self):
        return f"Change tool to " + self.desc

class RotateWorkpiece(Operation):
    def __init__(self, state, rotangle, label):
        self.rotangle = rotangle
        self.label = label
        super().__init__(state)

    def __next__(self):
        self.state.model = self.state.model.rotate(self.rotangle)
        self.state.material = self.state.material.rotate(self.rotangle)
        state.refresh_polyscape()
        self.done = True
        raise StopIteration

    def __str__(self):
        return f"Rotate {self.rotangle} degrees"	

class LinearInterpolate(Operation):
    def __init__(self, state, to, axesmask):
        self.to = to
        self.axesmask = axesmask
        super().__init__(state)
    
    def __next__(self):
        tomask = np.where(self.axesmask > 0.5, self.to, self.state.loc)
        self.state.material -= self.state.tool.generate_linear_cutarea(self.state.loc, tomask)
        self.state.loc = tomask
        state.refresh_polyscape()
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
            operations.append(ToolChange(state, Tool("endmill", tooldia*1e-3, 50e-3), f"{tooldia:.1f} mm endmill"))
            play = True
        psim.SameLine()
        if(psim.Button("Ball Nose")):
            operations.append(ToolChange(state, Tool("ballnose", tooldia*1e-3, 50e-3), f"{tooldia:.1f} mm ballnose"))
            play = True
        psim.SameLine()
        if(psim.Button("Drill")):
            operations.append(ToolChange(state, Tool("drill", tooldia*1e-3, 50e-3), f"{tooldia:.1f} mm drill"))
            play = True
        psim.TreePop()
    
    if (psim.TreeNode("Rotate Workpiece")):
        changed, rotangle = psim.InputFloat("Angle [deg]", rotangle) 
        psim.TextUnformatted("Rotation Axis")
        psim.SameLine()
        if(psim.Button("X")):
            operations.append(RotateWorkpiece(state, [rotangle, 0, 0], "X"))
            play = True
        psim.SameLine()
        if(psim.Button("Y")):
            operations.append(RotateWorkpiece(state, [0, rotangle, 0], "Y"))
            play = True
        psim.SameLine()
        if(psim.Button("Z")):
            operations.append(RotateWorkpiece(state, [0, 0, rotangle], "Z"))
            play = True
        psim.TreePop()
    
    if (psim.TreeNode("Linear Interpolate")):
        changed, cx = psim.Checkbox("X axis", cx) 
        if cx:
            psim.SameLine()
            changed, lx = psim.InputFloat("To X [mm]", lx) 
        changed, cy = psim.Checkbox("Y axis", cy)
        if cy:
            psim.SameLine()
            changed, ly = psim.InputFloat("To Y [mm]", ly)
        changed, cz = psim.Checkbox("Z axis", cz)
        if cz:
            psim.SameLine()
            changed, lz = psim.InputFloat("To Z [mm]", lz)
        if(psim.Button("Add")):
            operations.append(LinearInterpolate(state, np.array([lx*1e-3, ly*1e-3, lz*1e-3]), np.array([1 if cx else 0, 1 if cy else 0, 1 if cz else 0])))
            play = True
        psim.TreePop()
    
    psim.TextUnformatted("Playback")
    psim.SameLine()
    
    if (psim.Button("< Back")) and len(operations) > 0:
        if operationiter is not None:
            operationiter = None
        else:
            operationid = max(0, operationid - 1)
        state = operations[operationid].restore()
        lx, ly, lz = state.loc*1e3

    psim.SameLine()
    if (psim.Button("Delete")) and 0 <= operationid < len(operations):
        if operationiter is not None:
            operationiter = None
        state = operations[operationid].restore()
        lx, ly, lz = state.loc*1e3
        operations.pop(operationid)

    psim.SameLine()
    if (play or psim.Button("Play >")) and 0 <= operationid < len(operations):
        if operationiter is None:
            operationiter = iter(operations[operationid])
    
    if operationiter is not None:
        try:
            state = next(operationiter).state
        except StopIteration:
            operationiter = None
            operationid += 1
        lx, ly, lz = state.loc*1e3
    psim.ListBox("Operations", operationid, [str(o) for o in operations])	
    

model_file = 'ledmain.obj'
blank_size = np.array([75e-3, 25e-3, 50e-3])
state = State.generate(model_file, blank_size)
state.loc = np.array([0, 0, 50e-3])
state.init_polyscope()
