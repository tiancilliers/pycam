import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import time
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
import math
import sys
from copy import deepcopy
from scipy.spatial import KDTree
from manifold3d import Manifold, set_circular_segments, CrossSection, Mesh, JoinType
import pyclipr
import matplotlib.pyplot as plt
from multiprocessing import Pool

# TODO
# - smart path ordering to improve speed            DONE
# - detect arcs and interpolate                     DONE
# - detect heights for moving between operations    DONE
# - leadin
# - ballnose
# - pick faces to go to
# - custom feedrates                                DONE
# - violation detection
# - pick holes, flatten
# - optimization                                    PARTIALLY DONE
# - put G90.1 first (preamble)                      DONE
# - change zero position
# - only remove inside material/avoid clamps

def loadobj(filename):
    model = mm.loadMesh(filename)
    model.transform(mm.AffineXf3f.translation(-model.getBoundingBox().center()))
    mesh = Mesh(vert_properties = mn.getNumpyVerts(model), tri_verts = mn.getNumpyFaces(model.topology))   
    return Manifold(mesh)

class BooleanTree():
    def __init__(self, n):
        self.n = 2**math.ceil(math.log2(n))
        self.tree = [False] * (2*self.n-1)

    def get_sum(self, l, r):
        l += self.n - 1
        r += self.n - 2
        sum = 0
        while (l <= r):
            if ((l % 2) == 0):
                sum = sum or self.tree[l]
                l = (l + 1 - 1) // 2
            else:
                l = (l - 1) // 2
            if ((r % 2) == 1):
                sum = sum or self.tree[r]
                r = (r - 1 - 2) // 2
            else:
                r = (r - 2) // 2
        return sum

    def update(self, i, value):
        node = self.n - 1 + i
        self.tree[node] = value
        while node > 0:
            node = (node - 1) // 2
            left_child = node * 2 + 1
            right_child = node * 2 + 2
            self.tree[node] = self.tree[left_child] or self.tree[right_child]

class Tool:
    def __init__(self, type, diameter, length=50e-3, teeth=2, fpt=0.025):
        self.type = type
        self.diameter = diameter
        self.length = length
        self.teeth = teeth
        self.fpt = fpt
    
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
        if self.type == "drill":
            raise NotImplementedError("Drill not implemented")

        cutarea = cutarea.rotate(np.array([0, 0, -np.rad2deg(phi)])).translate(loc1)

        return cutarea
    
    def generate_curve_cutarea(self, loc1, loc2, center, code):
        #print(loc1, loc2, center, code)
        r = np.linalg.norm(loc1[:2]-center)
        angle = np.arctan2((-1 if code==3 else 1)*np.cross(loc2[:2]-center, loc1[:2]-center), np.dot(loc2[:2]-center, loc1[:2]-center)) % (2*np.pi)
        if np.linalg.norm(loc2[:2]-loc1[:2]) < 1e-9:
            angle = 2*np.pi
        angles = np.linspace(0, angle, math.ceil(abs(angle)/np.pi*32)+2)
        angles = (-angles if code==2 else angles) + np.arctan2(loc1[1]-center[1], loc1[0]-center[0])
        pts = np.array([[center[0]+r*np.cos(a), center[1]+r*np.sin(a)] for a in angles])
        pts = np.hstack((pts, np.linspace(loc1[2], loc2[2], len(pts)).reshape(-1,1)))
        #print(r, angle, angles, pts)
        #print()
        cutarea = Manifold()
        for i in range(len(pts)-1):
            cutarea += self.generate_linear_cutarea(pts[i], pts[i+1])
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

    def restore(self):
        return self.prev_state

class LinearInterpolate(Operation):
    def __init__(self, to, fast=False):
        self.to = to
        self.fast = fast
    
    def execute(self, state):
        super().execute(state)
        tomask = np.array([state.loc[i] if self.to[i] is None else self.to[i] for i in range(3)])
        state.material -= state.tool.generate_linear_cutarea(state.loc, tomask)
        state.material = state.material.as_original()
        state.loc = tomask
        if not self.fast:
            feedr = state.tool.teeth*state.tool.fpt*rpm*(1 if self.to[2] is None else 0.5)
            yield "G01 " + ' '.join('XYZ'[i] + f'{(self.to[i]*1e3):.03f}' for i in range(3) if self.to[i] is not None) + f" F{int(feedr)}"
        else:
            yield "G00 " + ' '.join('XYZ'[i] + f'{(self.to[i]*1e3):.03f}' for i in range(3) if self.to[i] is not None)
    def __str__(self):
        return f"Linear interpolate to {self.to}"


class CircularInterpolate(Operation):
    def __init__(self, to, center, code):
        self.to = to
        self.center = center
        self.code = code
    
    def execute(self, state):
        super().execute(state)
        tomask = np.array([state.loc[i] if self.to[i] is None else self.to[i] for i in range(3)])
        state.material -= state.tool.generate_curve_cutarea(state.loc, tomask, self.center, self.code)
        state.material = state.material.as_original()
        state.loc = tomask
        feedr = state.tool.teeth*state.tool.fpt*rpm*(1 if self.to[2] is None else 0.5)
        yield f"G0{self.code} " + ' '.join('XYZ'[i] + f'{(self.to[i]*1e3):.03f}' for i in range(3) if self.to[i] is not None) + ' ' + ' '.join('IJ'[i] + f'{(self.center[i]*1e3):.03f}' for i in range(2)) + f" F{int(feedr)}"
    
    def __str__(self):
        return f"Circular interpolate to {self.to}"

class ToolChange(Operation):
    def __init__(self, tool):
        self.tool = tool

    def execute(self, state):
        super().execute(state)
        yield from LinearInterpolate([None,None,50e-3]).execute(state)
        yield from LinearInterpolate([0,0,None]).execute(state)
        state.tool = self.tool
        yield "M5\nM0\n% CHANGE TOOL TO " + str(self.tool).upper() + "\nM3"

    def __str__(self):
        return f"Change tool to " + str(self.tool)

class RotateWorkpiece(Operation):
    def __init__(self, rotangle):
        self.rotangle = rotangle

    def execute(self, state):
        super().execute(state)
        yield from LinearInterpolate([None,None,50e-3]).execute(state)
        yield from LinearInterpolate([0,0,None]).execute(state)
        state.model = state.model.rotate(self.rotangle)
        state.material = state.material.rotate(self.rotangle)
        yield f"M5\nM0\n% ROTATE WORKPIECE BY {self.rotangle} DEGREES\nM3"

    def __str__(self):
        return f"Rotate {self.rotangle} degrees"	

def safe_offset(crosssec, delta, sign=1):
    fixed_polys = []
    for comp in crosssec.decompose():
        polys = crosssec.to_polygons()
        for poly in polys:
            n = len(poly)
            number = sum((poly[i%n][0]-poly[(i-1)%n][0])*(poly[i%n][1]+poly[(i-1)%n][1]) for i in range(n))
            if sign*number*delta < 0 or abs(number/2/(np.pi*delta**2)) > 1:
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
            if len(openPathsC[i]) > 0 and len(openPathsC[j]) > 0 and np.linalg.norm(openPathsC[i][0]-openPathsC[j][-1]) < 1e-6:
                openPathsC[i] = np.concatenate((openPathsC[j], openPathsC[i]))
                openPathsC[j] = np.array([])
    openPathsC = [path[::-1] if climb else path for path in openPathsC if len(path) > 0]

    return paths ^ bounds, openPathsC

def arcify_path(path):
    segs = []
    ii = 0
    jj = 4
    while jj < len(path):
        if jj-ii > 3:
            patharr = np.array(path[ii:jj+1])
            a = 2*np.sum((patharr[1:-1]-np.array(path[ii]))*np.array([path[ii][1]-path[jj][1], path[jj][0]-path[ii][0]]), axis=1).reshape(-1,1)
            b = np.array([path[k][0]*(path[k][0]-path[ii][0]-path[jj][0]) + path[k][1]*(path[k][1]-path[ii][1]-path[jj][1]) + path[ii][0]*path[jj][0] + path[ii][1]*path[jj][1] for k in range(ii+1, jj)])
            k = np.linalg.lstsq(a, b, rcond=None)[0][0]
            x, y = 0.5*(path[ii][0]+path[jj][0])+k*(path[ii][1]-path[jj][1]), 0.5*(path[ii][1]+path[jj][1])+k*(path[jj][0]-path[ii][0])
            p = np.array([x, y])
            r = np.linalg.norm(path[ii]-p)
            maxerr = np.max(np.abs(np.linalg.norm(patharr[1:-1]-p, axis=1)-r))
            circ = np.sum(np.linalg.norm(patharr[1:]-patharr[:-1], axis=1))
            maxerr = max(maxerr, np.max(np.abs(np.linalg.norm(0.5*patharr[1:]+0.5*patharr[:-1]-p, axis=1)-r)))
            if maxerr < 2e-5:
                if circ/r > 2*np.pi/64:
                    segs.append((jj-ii, ii, p))
                jj += 1
            else:
                ii += 1
        else:
            jj += 1
    used = BooleanTree(len(path))
    segs2 = []
    for seg in sorted(segs, key=lambda x: x[0], reverse=True):
        if not used.get_sum(seg[1], seg[1]+seg[0]):
            for k in range(seg[1]+1, seg[1]+seg[0]-1):
                used.update(k, True)
            segs2.append(seg)   
    segs3 = []
    segs2.sort(key=lambda x: x[1])
    #print(segs2)
    if len(segs2) == 0:
        return list(zip([1]*(len(path)-1), path[:-1], path[1:]))
    elif segs2[0][1] > 0:
        segs3 += list(zip([1]*segs2[0][1], path[:segs2[0][1]], path[1:segs2[0][1]+1]))
    for segi, seg in enumerate(segs2):
        code = 3 if np.cross(path[seg[1]]-seg[2], path[seg[1]+2]-seg[2]) > 0 else 2
        segs3.append((code, path[seg[1]], path[seg[1]+seg[0]], seg[2]))
        endidx = seg[1]+seg[0]
        nextstart = segs2[segi+1][1] if segi+1 < len(segs2) else len(path)-1
        if endidx < nextstart:
            #print([1]*(nextstart-endidx), path[endidx:nextstart-1], path[endidx+1:nextstart])
            segs3 += list(zip([1]*(nextstart-endidx), path[endidx:nextstart], path[endidx+1:nextstart+1]))
    return segs3

def find_topzstop(model, minz, maxz):
    cut = model.trim_by_plane([0,0,1], minz+1e-8).trim_by_plane([0,0,-1], -maxz+1e-8)
    verts = cut.to_mesh().vert_properties[:,:3]
    pts = []
    for tri in cut.to_mesh().tri_verts:
        zs = verts[tri,2]
        if max(zs)-min(zs) < 1e-9 and max(zs) > minz+2e-8 and min(zs) < maxz-2e-8:
            x1, y1, x2, y2, x3, y3 = *verts[tri[0],:2], *verts[tri[1],:2], *verts[tri[2],:2]
            norm = np.cross(verts[tri[1],:2]-verts[tri[0],:2], verts[tri[2],:2]-verts[tri[0],:2])
            area = -0.5*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
            if abs(area) > 1e-12 and norm > 0:
                pts.append([zs[0], area])
    pts.sort(key=lambda x: x[1])
    zstops = []
    for pt in pts:
        if not any(abs(zstop-pt[0]) < 1e-9 for zstop in zstops):
            zstops.append(pt[0])
    zstops = sorted(zstops)[::-1]
    return zstops[0] if len(zstops) > 0 else None

class RoughCut(Operation):
    def __init__(self, endz, stepz, stock=0.1e-3):
        self.endz = endz
        self.stepz = stepz
        self.stock = stock
    
    def path_stacks(state, botz, climb=True, stock=0.1e-3):
        bounds2 = state.material.trim_by_plane([0,0,1], botz+2e-8).project()
        bounds = safe_offset(bounds2, (0.5*state.tool.diameter)*(1-3*sf))
        cut = state.model.trim_by_plane([0,0,1], botz+2e-8).project()

        #plt.figure()
        #plt.xlim(-0.05, 0.05)
        #plt.ylim(-0.05, 0.05)
        #plt.gca().set_aspect('equal', adjustable='box')
        #for poly in cut.to_polygons():
        #    plt.plot([pt[0] for pt in poly], [pt[1] for pt in poly], "r-")
        #for poly in bounds.to_polygons():
        #    plt.plot([pt[0] for pt in poly], [pt[1] for pt in poly], "b-")
        cut = safe_offset(cut, stock)

        # offset outline until no operations remain
        cncpaths = []
        
        if cut.is_empty():
            paths, cncpath = process(bounds2, bounds, climb)
            while not len(cncpath) == 0:
                cncpaths.insert(0, cncpath[::-1])
                paths, cncpath = process(safe_offset(paths, -0.5*state.tool.diameter), bounds, climb)
        else:
            paths, cncpath = process(safe_offset(cut, 0.5*state.tool.diameter), bounds, climb)
            cncpaths.append(cncpath)
            while not (bounds - paths).is_empty():
                paths, cncpath = process(safe_offset(paths, 0.5*state.tool.diameter), bounds, climb)
                if len(cncpath) == 0:
                    break
                cncpaths.append(cncpath)
        
        # map paths to which paths need to be cut first
        descendants = [[[] for _ in range(len(layer))] for layer in cncpaths]
        for descarr, prevlayer, layer in zip(descendants, cncpaths[:-1], cncpaths[1:]):
            coords = np.concatenate([path for path in layer])
            idxs = [pathi for pathi in range(len(layer)) for _ in layer[pathi]]
            tree = KDTree(coords)
            for i, otherpath in enumerate(prevlayer):
                neighbors = tree.query_ball_point(otherpath, 0.51*state.tool.diameter)
                for k in set(idxs[i] for n in neighbors for i in n):
                    descarr[i].append(k)
        #print(descendants)
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
        #for path in process_arr:
        #    plt.plot(path[:,0], path[:,1], "k-")
        #    plt.plot(path[0,0], path[0,1], "k+")
        #for pp, xp in zip(process_arr[:-1], process_arr[1:]):
        #    plt.plot([pp[-1,0], xp[0,0]], [pp[-1,1], xp[0,1]], "k--")
        
        #segs = [arcify_path(path) for path in process_arr]
        segs = pool.map(arcify_path, process_arr)
        #for path in segs:
        #    for seg in path:
        #        if seg[0] == 1:
        #            plt.plot([seg[1][0], seg[2][0]], [seg[1][1], seg[2][1]], "b-")
        #        elif seg[0] == 2:
        #            plt.plot([seg[1][0], seg[3][0]], [seg[1][1], seg[3][1]], "r-")
        #            plt.plot([seg[2][0], seg[3][0]], [seg[2][1], seg[3][1]], "r-")
        #        elif seg[0] == 3:
        #            plt.plot([seg[1][0], seg[3][0]], [seg[1][1], seg[3][1]], "g-")
        #            plt.plot([seg[2][0], seg[3][0]], [seg[2][1], seg[3][1]], "g-")
        #print(segs)

        #plt.show()
        return segs

    def execute(self, state):
        super().execute(state)
        yield from LinearInterpolate([None,None,50e-3]).execute(state)

        cut = state.model.trim_by_plane([0,0,1], self.endz+self.stock)
        startz = cut.bounding_box()[5]
        nslices = math.ceil((startz-self.endz-self.stock)/self.stepz)
        for botz in [self.endz + i*self.stepz for i in range(nslices)][::-1]:
            cutz = botz
            while cutz is not None:
                slice_path = RoughCut.path_stacks(state, cutz+self.stock, stock=self.stock)
                for pi,path in enumerate(slice_path):
                    yield from LinearInterpolate([path[0][1][0],path[0][1][1],None], fast=True).execute(state)
                    yield from LinearInterpolate([None,None,cutz+self.stock]).execute(state)
                    for seg in path:
                        if seg[0] == 1:
                            yield from LinearInterpolate([seg[2][0],seg[2][1],None]).execute(state)
                        else:
                            yield from CircularInterpolate([seg[2][0],seg[2][1],None], seg[3], seg[0]).execute(state)
                    if pi < len(slice_path)-1:
                        endpt = path[-1][2]
                        nextstart = slice_path[pi+1][0][1]
                        highest = (state.material ^ state.tool.generate_linear_cutarea(np.pad(endpt, (0,1), 'constant'), np.pad(nextstart, (0,1), 'constant'))).bounding_box()[5]+2e-3
                        yield from LinearInterpolate([None,None,highest], fast=True).execute(state)
                    else:
                        yield from LinearInterpolate([None,None,state.material.bounding_box()[5]+2e-3], fast=True).execute(state)
                cutz = find_topzstop(state.model, botz+self.stock, botz+self.stepz+self.stock) if abs(cutz-botz)<1e-9 else find_topzstop(state.model, botz+self.stock, cutz)
    
    def __str__(self):
        return f"Rough cut from {self.endz} steps of {self.stepz} with {self.stock} stock"

def polyscope_callback():
    global operations, opcodes, operationid, subiter, subid, tooldia, rotangle, state, lx, ly, lz, cx, cy, cz, tz, sz, play, lvs, teeth, fpt

    if (psim.TreeNode("Change Tool")):
        changed, tooldia = psim.InputFloat("Diameter [mm]", tooldia) 
        changed, teeth = psim.InputFloat("Teeth []", teeth) 
        changed, fpt = psim.InputFloat("Feed per tooth [mm]", fpt) 
        psim.TextUnformatted("Tool Type")
        psim.SameLine()
        if(psim.Button("End Mill")):
            operations.append((ToolChange(Tool("endmill", tooldia*1e-3, 50e-3, teeth, fpt)), []))
        psim.SameLine()
        if(psim.Button("Ball Nose")):
            operations.append((ToolChange(Tool("ballnose", tooldia*1e-3, 50e-3, teeth, fpt)), []))
        psim.SameLine()
        if(psim.Button("Drill")):
            operations.append((ToolChange(Tool("drill", tooldia*1e-3, 50e-3, teeth, fpt)), []))
        psim.TreePop()
    
    if (psim.TreeNode("Rotate Workpiece")):
        changed, rotangle = psim.InputFloat("Angle [deg]", rotangle) 
        psim.TextUnformatted("Rotation Axis")
        psim.SameLine()
        if(psim.Button("X")):
            operations.append((RotateWorkpiece([rotangle, 0, 0]), []))
        psim.SameLine()
        if(psim.Button("Y")):
            operations.append((RotateWorkpiece([0, rotangle, 0]), []))
        psim.SameLine()
        if(psim.Button("Z")):
            operations.append((RotateWorkpiece([0, 0, rotangle]), []))
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
            operations.append((LinearInterpolate([lx*1e-3 if cx else None, ly*1e-3 if cy else None, lz*1e-3 if cz else None]), []))
            play = True
        psim.TreePop()
    
    if (psim.TreeNode("Rough Cut")):
        changed, tz = psim.InputFloat("Downto Z [mm]", tz)
        changed, sz = psim.InputFloat("Step Z [mm]", sz)
        changed, lvs = psim.InputFloat("Leave Stock [mm]", lvs)
        if(psim.Button("Add")):
            operations.append((RoughCut(tz*1e-3, sz*1e-3, stock=lvs*1e-3), []))
        psim.TreePop()
    
    psim.TextUnformatted("Playback")
    psim.SameLine()
    
    if (psim.Button("< Back")) and operationid > 0:
        play = False
        if subiter is None:
            operationid -= 1
        else:
            subiter = None
        state = operations[operationid][0].restore()
        operations[operationid][1].clear()
        lx, ly, lz = state.loc*1e3
        state.refresh_polyscape()

    psim.SameLine()
    if (psim.Button("Delete Last")) and operationid == len(operations):
        operationid -= 1
        state = operations[operationid][0].restore()
        lx, ly, lz = state.loc*1e3
        state.refresh_polyscape()
        operations.pop(operationid)

    psim.SameLine()
    if (psim.Button("Process >") or play) and 0 <= operationid < len(operations):
        play = True

    psim.SameLine()
    if (psim.Button("Step single") or play) and 0 <= operationid < len(operations):
        if subiter is None:
            subiter = operations[operationid][0].execute(state)
        try:
            operations[operationid][1].append(next(subiter))
        except StopIteration:
            subiter = None
            operationid += 1
            play = False
        lx, ly, lz = state.loc*1e3
        state.refresh_polyscape()
    
    psim.ListBox("Operations", operationid, [str(o[0]) for o in operations])
    gcodes = [f"G0 G90.1 G54 G17 G40 G49 G80\nS{rpm} M3"] + [desc for o in operations for desc in o[1]] + ["M30"]
    psim.ListBox("G-code", len(gcodes)-1, gcodes)
    if psim.Button("SAVE GCODE"):
        with open("gcode.nc", "w") as f:
            f.write("\n".join(gcodes))

if __name__ == "__main__":
    pool = Pool()
    cs = 32
    rpm = 10000
    set_circular_segments(cs)
    sf = 1-np.cos(np.pi/cs)
    #print(3*sf)
    Manifold.__deepcopy__ = lambda self, memo: Manifold.compose([self])

    operations = []
    opcodes = []
    operationid = 0
    subiter = None
    subid = 0

    tooldia = 4
    rotangle = 90
    lx, ly, lz = 0, 0, 50
    cx, cy, cz = True, True, True
    tz, sz = 8.5, 4
    play = False
    lvs = 0.1
    teeth, fpt = 2.0, 0.025

    model_file = 'ledmain.obj'
    blank_size = np.array([75e-3, 25e-3, 50e-3])
    state = State.generate(model_file, blank_size)
    state.loc = np.array([0, 0, 50e-3])
    state.init_polyscope()