import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import time
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
import math
import sys
from copy import deepcopy
from manifold3d import Manifold, set_circular_segments, CrossSection, Mesh, JoinType
import pyclipr
set_circular_segments(32)

# TODO
# - detect arcs and interpolate
# - detect heights for moving between operations
# - leadin
# - ballnose
# - pick faces to go to
# - custom feedrates
# - violation detection
# - pick holes, flatten
# - optimization

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

    def __str__(self):
        return f"{self.diameter:.1f} mm {self.type}"

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
    def execute(self, state):
        self.prev_state = deepcopy(state)
        for op, desc in self.steps():
            op(state)

    def steps(self):
        return
    
    def restore(self):
        return self.prev_state

class LinearInterpolate(Operation):
    def __init__(self, to):
        self.to = to
    
    def steps(self):
        def g01(state):
            tomask = np.array([state.loc[i] if self.to[i] is None else self.to[i] for i in range(3)])
            state.material -= state.tool.generate_linear_cutarea(state.loc, tomask)
            state.loc = tomask
        yield (g01, "G01 " + ' '.join('XYZ'[i] + f'{self.to[i]:.03f}' for i in range(3) if self.to[i] is not None))

    def __str__(self):
        return f"Linear interpolate to {self.to}"

class ToolChange(Operation):
    def __init__(self, tool):
        self.tool = tool

    def steps(self):
        yield from LinearInterpolate([None,None,50e-3]).steps()
        yield from LinearInterpolate([0,0,None]).steps()
        def change(state):
            state.tool = self.tool
        yield (change, "% CHANGE TOOL TO " + str(self.tool).upper())

    def __str__(self):
        return f"Change tool to " + str(self.tool)

class RotateWorkpiece(Operation):
    def __init__(self, rotangle):
        self.rotangle = rotangle

    def steps(self):
        yield from LinearInterpolate([None,None,50e-3]).steps()
        yield from LinearInterpolate([0,0,None]).steps()
        def rotate(state):
            state.model = state.model.rotate(self.rotangle)
            state.material = state.material.rotate(self.rotangle)
        yield (rotate, f"% ROTATE WORKPIECE BY {self.rotangle} DEGREES")

    def __str__(self):
        return f"Rotate {self.rotangle} degrees"	

class SurfaceFlatten(Operation):
    def __init__(self, startz, endz, stepz, xe, ye, stepover):
        self.startz = startz
        self.endz = endz
        self.stepz = stepz
        self.xe = xe
        self.ye = ye
        self.stepover = stepover
    
    def steps(self):
        for z in np.arange(self.startz, self.endz, self.stepz):
            x = -self.xe
            yield from LinearInterpolate([None,None,50e-3]).steps()
            yield from LinearInterpolate([x, -self.ye, None]).steps()
            yield from LinearInterpolate([None,None,z]).steps()
            while x <= self.xe:
                yield from LinearInterpolate([x, self.ye, None]).steps()
                x += self.stepover
                yield from LinearInterpolate([x, self.ye, None]).steps()
                yield from LinearInterpolate([x, -self.ye, None]).steps()
                x += self.stepover
                yield from LinearInterpolate([x, -self.ye, None]).steps()
    
    def __str__(self):
        return f"Surface flatten from {self.startz} to {self.endz} with step {self.stepz}"

def safe_offset(crosssec, delta):
    fixed_polys = []
    for comp in crosssec.decompose():
        polys = crosssec.to_polygons()
        nums = []
        for poly in polys:
            n = len(poly)
            number = sum((poly[i%n][0]-poly[(i-1)%n][0])*(poly[i%n][1]+poly[(i-1)%n][1]) for i in range(n))
            nums.append(number)
            print(number, delta, number/2, np.pi*delta**2/2)
            if number*delta < 0 or number/2/(np.pi*delta**2) > 1:
                fixed_polys.append(poly)
    cleaned = CrossSection(fixed_polys)
    return cleaned.offset(delta, JoinType.Round).simplify()

def process(paths, bounds, climb=True):
    pc2 = pyclipr.Clipper()
    pc2.scaleFactor = int(1e6)
    pc2.addPaths([np.concatenate((poly,poly[0:1,:])) for poly in paths.to_polygons()], pyclipr.Subject, True)
    pc2.addPaths(bounds.to_polygons(), pyclipr.Clip, False)
    _ = pc2.execute(pyclipr.Intersection, pyclipr.FillRule.NonZero)
    _, openPathsC = pc2.execute(pyclipr.Intersection, pyclipr.FillRule.NonZero, returnOpenPaths=True)
    for i in range(len(openPathsC)):
        for j in range(i+1, len(openPathsC)):
            if len(openPathsC[j]) > 0 and np.linalg.norm(openPathsC[i][0]-openPathsC[j][-1]) < 1e-6:
                openPathsC[i] = np.concatenate((openPathsC[j], openPathsC[i]))
                openPathsC[j] = np.array([])
    openPathsC = [path[::-1] if climb else path for path in openPathsC if len(path) > 0]
    return paths ^ bounds, openPathsC

class RoughCut(Operation):
    def __init__(self, state, startz, endz, stepz):
        nslices = math.ceil((startz-endz)/stepz)
        self.slices = [endz + i*stepz for i in range(nslices)][::-1]
        self.slice_paths = [RoughCut.path_stacks(state, botz, stepz) for botz in self.slices]
    
    def path_stacks(state, botz, stepz, climb=True, stock=0.1e-3):
        bounds = state.material.trim_by_plane([0,0,1], botz).trim_by_plane([0,0,-1], -botz-stepz).project()
        bounds = safe_offset(bounds, 0.5*state.tool.diameter)
        cut = state.model.trim_by_plane([0,0,1], botz).trim_by_plane([0,0,-1], -botz-stepz).project() ^ bounds
        cut = safe_offset(cut, stock)

        # offset outline until no operations remain
        cncpaths = []
        paths, cncpath = process(safe_offset(cut, 0.5*state.tool.diameter) if not cut.is_empty() else CrossSection.circle(0.4*state.tool.diameter), bounds, climb)
        cncpaths.append(cncpath)
        while not (bounds - paths).is_empty():
            paths, cncpath = process(safe_offset(paths, 0.5*state.tool.diameter), bounds, climb)
            if len(cncpath) == 0:
                break
            cncpaths.append(cncpath)
        
        # map paths to which paths need to be cut first
        descendants = [[[] for _ in range(len(layer))] for layer in cncpaths]
        for descarr, prevlayer, layer in zip(descendants, cncpaths[:-1], cncpaths[1:]):
            for pathi, path in enumerate(layer):
                mintotaldist, pathidx = 1e309, -1
                for i, otherpath in enumerate(prevlayer):
                    totaldist = sum(np.linalg.norm(point-otherpath, ord=2, axis=1).min() for point in path)
                    if totaldist < mintotaldist:
                        mintotaldist, pathidx = totaldist, i
                if pathidx != -1:
                    descarr[pathidx].append(pathi)
        active = [(i,j) for i in range(len(cncpaths)) for j in range(len(cncpaths[i])) if len(descendants[i][j]) == 0]
        process_arr = []

        # sort paths to minimize travel distance
        while len(active) > 0:
            minidx = 0 if len(process_arr) == 0 else min(range(len(active)), key=lambda x: np.linalg.norm(process_arr[-1][-1,:]-cncpaths[active[x][0]][active[x][1]][0,:]))
            i,j = active.pop(minidx)
            process_arr.append(cncpaths[i][j])
            for nj, arr in enumerate(descendants[i-1]):
                if j in arr:
                    arr.remove(j)
                    if len(arr) == 0:
                        active.append((i-1, nj))
        return process_arr

    def steps(self):
        yield from LinearInterpolate([None,None,50e-3]).steps()
        for botz, slice_path in zip(self.slices, self.slice_paths):
            for path in slice_path:
                yield from LinearInterpolate([path[0][0],path[0][1],None]).steps()
                yield from LinearInterpolate([None,None,botz]).steps()
                for point in path[1:]:
                    yield from LinearInterpolate([point[0],point[1],None]).steps()
                yield from LinearInterpolate([None,None,50e-3]).steps()
    
    def __str__(self):
        return f"Rough cut from {self.slices[-1]} to {self.slices[0]}"

operations = []
operationid = 0
subiter = None
subid = 0

tooldia = 4
rotangle = 90
lx, ly, lz = 0, 0, 50
cx, cy, cz = True, True, True
fz, tz, sz = 50, 0, 1
play = False

def polyscope_callback():
    global operations, operationid, subiter, subid, tooldia, rotangle, state, lx, ly, lz, cx, cy, cz, fz, tz, sz, play

    if (psim.TreeNode("Change Tool")):
        changed, tooldia = psim.InputFloat("Diameter [mm]", tooldia) 
        psim.TextUnformatted("Tool Type")
        psim.SameLine()
        if(psim.Button("End Mill")):
            operations.append(ToolChange(Tool("endmill", tooldia*1e-3, 50e-3)))
        psim.SameLine()
        if(psim.Button("Ball Nose")):
            operations.append(ToolChange(Tool("ballnose", tooldia*1e-3, 50e-3)))
        psim.SameLine()
        if(psim.Button("Drill")):
            operations.append(ToolChange(Tool("drill", tooldia*1e-3, 50e-3)))
        psim.TreePop()
    
    if (psim.TreeNode("Rotate Workpiece")):
        changed, rotangle = psim.InputFloat("Angle [deg]", rotangle) 
        psim.TextUnformatted("Rotation Axis")
        psim.SameLine()
        if(psim.Button("X")):
            operations.append(RotateWorkpiece([rotangle, 0, 0]))
        psim.SameLine()
        if(psim.Button("Y")):
            operations.append(RotateWorkpiece([0, rotangle, 0]))
        psim.SameLine()
        if(psim.Button("Z")):
            operations.append(RotateWorkpiece([0, 0, rotangle]))
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
            operations.append(LinearInterpolate([lx*1e-3 if cx else None, ly*1e-3 if cy else None, lz*1e-3 if cz else None]))
            play = True
        psim.TreePop()
    
    if (psim.TreeNode("Rough Cut")):
        changed, fz = psim.InputFloat("From Z [mm]", fz)
        changed, tz = psim.InputFloat("Downto Z [mm]", tz)
        changed, sz = psim.InputFloat("Step Z [mm]", sz)
        if(psim.Button("Add")):
            operations.append(RoughCut(state, fz*1e-3, tz*1e-3, sz*1e-3))
        psim.TreePop()
    
    psim.TextUnformatted("Playback")
    psim.SameLine()
    
    if (psim.Button("< Back")) and operationid > 0:
        play = False
        if subiter is None:
            operationid -= 1
        else:
            subiter = None
            subid = 0
        state = operations[operationid].restore()
        lx, ly, lz = state.loc*1e3
        state.refresh_polyscape()

    psim.SameLine()
    if (psim.Button("Delete Last")) and operationid == len(operations):
        operationid -= 1
        state = operations[operationid].restore()
        lx, ly, lz = state.loc*1e3
        state.refresh_polyscape()
        operations.pop(operationid)

    psim.SameLine()
    if (psim.Button("Play >") or play) and 0 <= operationid < len(operations):
        play = True

    psim.SameLine()
    if (psim.Button("Step single") or play) and 0 <= operationid < len(operations):
        if subiter is None:
            subiter = operations[operationid].steps()
            subid = 0
            operations[operationid].prev_state = deepcopy(state)
        try:
            next(subiter)[0](state)
            subid += 1
        except StopIteration:
            subiter = None
            subid = 0
            operationid += 1
            play = False
        lx, ly, lz = state.loc*1e3
        state.refresh_polyscape()
    
    psim.ListBox("Operations", operationid, [str(o) for o in operations])
    if 0 <= operationid < len(operations):
        psim.ListBox("G-code", subid, [desc for op,desc in operations[operationid].steps()])
    else:
        psim.ListBox("G-code", 0, [])
    

model_file = 'ledmain.obj'
blank_size = np.array([75e-3, 25e-3, 50e-3])
state = State.generate(model_file, blank_size)
state.loc = np.array([0, 0, 50e-3])
state.init_polyscope()
