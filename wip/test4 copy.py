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


climb = True
stock = 0.1e-3


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

model_file = 'ledmain.obj'
blank_size = np.array([75e-3, 25e-3, 50e-3])
model = loadobj(model_file).rotate([90,0,0])
material = Manifold.cube(blank_size, True).rotate([90,0,0])
tool = Tool("endmill", 4e-3)

plt.xlim(-0.05, 0.05)
plt.ylim(-0.05, 0.05)
plt.gca().set_aspect('equal', adjustable='box')

for poly in model.trim_by_plane([0,0,1], 0e-3).trim_by_plane([0,0,-1], -5e-3).project() .to_polygons():
    plt.plot([pt[0] for pt in poly], [pt[1] for pt in poly], "r-")



bounds = material.trim_by_plane([0,0,1], 0e-3).trim_by_plane([0,0,-1], -5e-3).project()
bounds = safe_offset(bounds, 2e-3)

cut = model.trim_by_plane([0,0,1], 0e-3).trim_by_plane([0,0,-1], -5e-3).project() ^ bounds
cut = safe_offset(cut, stock)


def process(paths):
    pc2 = pyclipr.Clipper()
    pc2.scaleFactor = int(1e6)
    pc2.addPaths([np.concatenate((poly,poly[0:1,:])) for poly in paths.to_polygons()], pyclipr.Subject, True)
    pc2.addPaths(bounds.to_polygons(), pyclipr.Clip, False)
    _ = pc2.execute(pyclipr.Intersection, pyclipr.FillRule.NonZero)
    _, openPathsC = pc2.execute(pyclipr.Intersection, pyclipr.FillRule.NonZero, returnOpenPaths=True)
    for i in range(len(openPathsC)):
        for j in range(i+1, len(openPathsC)):
            if len(openPathsC[i]) > 0 and np.linalg.norm(openPathsC[i][0]-openPathsC[j][-1]) < 1e-6:
                openPathsC[i] = np.concatenate((openPathsC[j], openPathsC[i]))
                openPathsC[j] = np.array([])
    openPathsC = [path[::-1] if climb else path for path in openPathsC if len(path) > 0]
    return paths ^ bounds, openPathsC

cncpaths = []
paths, cncpath = process(safe_offset(cut, 2e-3) if not cut.is_empty() else CrossSection.circle(0.4*tool.diameter))
cncpaths.append(cncpath)
while not (bounds - paths).is_empty():
    paths, cncpath = process(safe_offset(paths, 2e-3))
    if len(cncpath) == 0:
        break
    cncpaths.append(cncpath)
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
while len(active) > 0:
    minidx = 0 if len(process_arr) == 0 else min(range(len(active)), key=lambda x: np.linalg.norm(process_arr[-1][-1,:]-cncpaths[active[x][0]][active[x][1]][0,:]))
    i,j = active.pop(minidx)
    process_arr.append(cncpaths[i][j])
    for nj, arr in enumerate(descendants[i-1]):
        if j in arr:
            arr.remove(j)
            if len(arr) == 0:
                active.append((i-1, nj))

for path in process_arr:
    plt.plot(path[:,0], path[:,1], "g-")
    plt.plot(path[0,0], path[0,1], "g+")
for pp, xp in zip(process_arr[:-1], process_arr[1:]):
    plt.plot([pp[-1,0], xp[0,0]], [pp[-1,1], xp[0,1]], "g--")

idx = 33
plt.plot(process_arr[idx][:,0], process_arr[idx][:,1], "b-+")
plt.plot(process_arr[idx][0,0], process_arr[idx][0,1], "b+")

def arcify_path(path, i, j):
    print("ap", i, j)
    ii = i
    jj = i+2
    longest = (2, ii, None)
    while jj < j:
        if jj-ii > 3:
            a = np.array([[2*((path[k][0]-path[ii][0])*(path[ii][1]-path[jj][1]) + (path[k][1]-path[ii][1])*(path[jj][0]-path[ii][0]))] for k in range(ii+1, jj)])
            b = np.array([path[k][0]*(path[k][0]-path[ii][0]-path[jj][0]) + path[k][1]*(path[k][1]-path[ii][1]-path[jj][1]) + path[ii][0]*path[jj][0] + path[ii][1]*path[jj][1] for k in range(ii+1, jj)])
            k = np.linalg.lstsq(a, b, rcond=None)[0][0]
            x, y = 0.5*(path[ii][0]+path[jj][0])+k*(path[ii][1]-path[jj][1]), 0.5*(path[ii][1]+path[jj][1])+k*(path[jj][0]-path[ii][0])
            p = np.array([x, y])
            r = np.linalg.norm(path[ii]-p)
            maxerr = max(abs(np.linalg.norm(path[k]-p)-r) for k in range(ii+1, jj))
            maxerr = max(maxerr, max(abs(np.linalg.norm(0.5*path[k]+0.5*path[k+1]-p)-r) for k in range(ii, jj)))
            if maxerr < 2e-5:
                if jj-ii > longest[0]:
                    longest = (jj-ii, ii, p)
                jj += 1
            else:
                ii += 1
        else:
            jj += 1
    if longest[2] is not None:
        code = 3 if np.cross(path[longest[1]+1]-path[longest[1]], longest[2]-path[longest[1]]) > 0 else 2
        return arcify_path(path, i, longest[1]) + [(code, path[longest[1]], path[longest[1]+longest[0]], longest[2])] + arcify_path(path, longest[1]+longest[0], j)
    else:
        return list(zip([1]*(j-i), path[i:j], path[i+1:j+1]))

#segs = arcify_path(process_arr[idx], 0, len(process_arr[idx]))
#for seg in segs:
#    if seg[0] == 1:
#        plt.plot([seg[1][0], seg[2][0]], [seg[1][1], seg[2][1]], "g-+")
#    else:
#        plt.plot([seg[1][0], seg[3][0], seg[2][0]], [seg[1][1], seg[3][1], seg[2][1]], "b-+")
#plt.show()