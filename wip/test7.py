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

cs = 32
set_circular_segments(cs)
sf = 1-np.cos(np.pi/cs)
#print(3*sf)
Manifold.__deepcopy__ = lambda self, memo: Manifold.compose([self])


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

        cuttool = self.generate_bit(np.zeros(3)).rotate(np.array([0, 0, np.rad2deg(phi)])).rotate(np.array([-np.rad2deg(theta), 0, 0]))

        verts = cuttool.to_mesh().vert_properties
        kdtree = KDTree(verts[:,:2])

        projected = cuttool.project()

        def warpfunc(v):
            # query kdtree
            idxs = kdtree.query_ball_point(v[:2], 1e-8)
            z = max(verts[idxs,2]) if v[2] <= 1e-8 else min(verts[idxs,2])+dxyz
            return [v[0], v[1], z]

        cutarea = Manifold.extrude(projected, 5e-3).warp(warpfunc).rotate(np.array([np.rad2deg(theta),0,0])).rotate(np.array([0, 0, -np.rad2deg(phi)])).translate(loc1)

        cutarea += self.generate_bit(loc1)
        cutarea += self.generate_bit(loc2)

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

tool = Tool("ballnose", 3e-3)
cutarea = tool.generate_linear_cutarea(np.array([1e-3,1e-3,0]), np.array([5e-3,-7e-3,0]))
tool1 = Tool("ballnose", 3e-3).generate_bit(np.array([1e-3,1e-3,0]))
tool2 = Tool("ballnose", 3e-3).generate_bit(np.array([5e-3,-7e-3,2e-3]))

ps.init()
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.register_surface_mesh("cutarea", cutarea.to_mesh().vert_properties[:,:3], cutarea.to_mesh().tri_verts)
ps.register_surface_mesh("tool1", tool1.to_mesh().vert_properties[:,:3], tool1.to_mesh().tri_verts)
ps.register_surface_mesh("tool2", tool2.to_mesh().vert_properties[:,:3], tool2.to_mesh().tri_verts)
ps.show()