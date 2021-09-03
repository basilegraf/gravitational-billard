#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 11:24:14 2021

@author: basile
"""

import sympy as sp
import numpy as np
from enum import Enum
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import copy

# Use qt for animations
try:
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt')
except:
    pass


# Force on mobile mass at p0 + v0 * t exerted by fixed mass at c
# assuming unit masses and gravitation constant
def force(p0, v0, c, t):
    p0 = np.asarray(p0)
    v0 = np.asarray(v0)
    c = np.asarray(c)
    x = c - (p0 + v0 * t)
    r = np.linalg.norm(x)
    return (1.0 / r**3) * x


# simulate trajectory deviation, assuming unit parameters
def simulate(p0, v0, c, T):
    p0 = np.asarray(p0)
    v0 = np.asarray(v0)
    c = np.asarray(c)
    T = np.asarray(T)
    z0 = np.array([0,0,0,0]);
    fInt = lambda tt, zz : np.concatenate((zz[2:4], force(p0, v0, c, tt)))
    tSpan = [0, T]
    if T.shape == ():
        tSpan = [0, T.tolist()]
        tEval = None
    else:
        tSpan = [T[0], T[-1]]
        tEval = T
    if tSpan[0] != tSpan[1]:
        ivpSol = integrate.solve_ivp(fInt, tSpan, z0, t_eval = tEval)
        return (ivpSol.t, ivpSol.y[0:2,:], ivpSol.y[2:4,:])
    else:
        pp = np.zeros((2,1))
        pp[:,0] = p0
        vv = np.zeros((2,1))
        vv[:,0] = v0
        return (np.zeros((1)), pp, vv)
    

p0 = np.asarray([1.0,1.0])
v0 = np.asarray([0.0,1.0])
c = np.asarray([0.0,0.0])
(t, dpos, dspd) = simulate(p0, v0, c,1)


pos = dpos + np.asarray([t * v0[0] + p0[0], t * v0[1] + p0[1]])


plt.plot(pos[0,:], pos[1,:])
plt.axis('equal')



G = 6.67e-11        # gravitation constant
M = 80.0            # player mass
c = [1.5, 0.0]      # player position

W = 1.3/2           # table half width
H = 2.5/2           # table half height
R = 60/1000/2       # ball radius



# Ball with initial conditions
class ball:
    def __init__(self, p0 = [0,0], v = [0,0], R = R, idx = -1):
        self.p0 = np.asarray(p0)
        self.v = np.asarray(v)
        self.R = np.abs(R)
        self.idx = idx
    
    def solution(self, T, gravityOn):
        eps = G * M if gravityOn else 0.0
        (t, dpos, dspd) = simulate(self.p0, self.v, c, T)
        pos = eps * dpos + np.asarray([t * self.v[0] + self.p0[0], t * self.v[1] + self.p0[1]])
        spd = np.asarray([self.v[0] + eps * dspd[0, :], self.v[1] + eps * dspd[1, :]])
        return (t, pos, spd)
    
    def propagate(self, T, gravityOn):
        (t, pos, spd) = self.solution(T, gravityOn)       
        self.p0 = pos[:,-1]
        self.v = spd[:,-1]
        return (t, pos, spd)
        
        
        

# collison time between two balls (no gravity)
def ballBallCollisonNoGravity(ball1, ball2):
    p1 = ball1.p0
    p2 = ball2.p0
    v1 = ball1.v
    v2 = ball2.v
    r = ball1.R + ball2.R
    d = p1 - p2
    w = v1 - v2
        
    A = np.dot(w, w)
    B = 2 * np.dot(d, w)
    C = np.dot(d, d) - r**2
    discr = B**2 - 4 * A * C
    
    if (A == 0.0):
        # no relative speed
        return (np.Inf, v1, v2)
    
    if (discr < 0):
        # no solution (no collision)
        return (np.Inf, v1, v2)

    t1 = (-B + np.sqrt(discr)) / (2 * A)
    t2 = (-B - np.sqrt(discr)) / (2 * A)
    # ball-distance time-derivative at solutions t1 and t2
    dist_dt_t1 = 2 * (np.dot(d, w) + np.dot(w, w) * t1)
    dist_dt_t2 = 2 * (np.dot(d, w) + np.dot(w, w) * t2)
    
    print("(%d,%d)-Distance derivatives d(%f) = %f, d(%f) = %f, norm(w) = %f" % (ball1.idx, ball2.idx, t1, dist_dt_t1, t2, dist_dt_t2, np.linalg.norm(w)))
    
    # collision is the solution with negative distance derivative
    t = np.Inf
    if (dist_dt_t1 < 0.0):
        t = t1
    elif (dist_dt_t2 < 0.0):
        t = t2
    else:
        raise Exception("Undefined collision, should not happen") 
        
    # Discard any time smaller than minus the time it takes to cover r/2
    # This allows small negative time when correcting a collision with gravity pull
    # However it discards collision solution at negative time _after_ an actual collision and speed reflexion
    tMin = - abs(t1 - t2) / 2
    if (t < tMin): 
        return (np.Inf, v1, v2)
    else:
        # Compute reflexion 
        pp1 = p1 + t * v1
        pp2 = p2 + t * v2
        d = pp1 - pp2
        d = d / np.linalg.norm(d)
        n = np.array([d[1], -d[0]])        

        A = np.array([n, d])
        
        vv1 = np.matmul(A, v1)
        vv2 = np.matmul(A, v2)
        
        tmp = vv1[1]
        vv1[1] = vv2[1]
        vv2[1] = tmp

        vvv1 = np.matmul(A.transpose(), vv1)
        vvv2 = np.matmul(A.transpose(), vv2)
        return (t, vvv1, vvv2)


def ballBallCollison(ball1, ball2, gravityOn):
    (t, v1, v2) = ballBallCollisonNoGravity(ball1, ball2)
    if t == np.Inf or not gravityOn:
        return (t, v1, v2) 
    # correction by computing new collision from corrected end-trajectory
    t1, pos1, spd1 = ball1.solution(t, True)
    t2, pos2, spd2 = ball2.solution(t, True)
    ball1New = ball(pos1[:,-1], spd1[:,-1], ball1.R, ball1.idx+10)
    ball2New = ball(pos2[:,-1], spd2[:,-1], ball2.R, ball2.idx+10)
    (tCorr, v1Corr, v2Corr) = ballBallCollisonNoGravity(ball1New, ball2New)
    return (t + tCorr, v1Corr, v2Corr)
    

# Collision between ball and band
def ballBandCollisionNoGravity(ball1):
    p = ball1.p0
    v = ball1.v
    if (np.linalg.norm(v) == 0.0):
        return (np.Inf, v)
    # Horizontal time
    tH = 0.0
    if v[0] == 0.0:
        tH = np.Inf
    else:
        tH = (W - p[0]) / v[0] if v[0] > 0.0 else (-W - p[0]) / v[0]
    # Verical time
    tV = 0.0
    if v[1] == 0.0:
        tV = np.Inf
    else:
        tV = (H - p[1]) / v[1] if v[1] > 0.0 else (-H - p[1]) / v[1]
    # Keep minimal time
    vv = v.copy()
    if tH < tV:
        vv[0] = -vv[0]
        return (tH, vv)
    else:
        vv[1] = -vv[1]
        return (tV, vv)
    
    
def ballBandCollision(ball1, gravityOn):
    (t, v) = ballBandCollisionNoGravity(ball1)
    if t == np.Inf or not gravityOn:
        return (t, v)
    # correction by computing new collision from corrected end-trajectory
    t1, pos1, spd1 = ball1.solution(t, True)
    ball1New = ball(pos1[:,-1], spd1[:,-1], ball1.R)
    (tCorr, vCorr) = ballBandCollisionNoGravity(ball1)
    return (t + tCorr, vCorr)
    
class billard:
    def __init__(self, gravityOn = False):
        self.gravityOn = gravityOn
        # balls
        self.balls = [ball([0.*R,-H/2], [0,2], R, 0)] # white ball
        n = 4
        d = 1.1 * 2 * R
        y = 0
        idx = 1
        for k in range(1,n+1):
            x0 = -d * (k - 1) / 2
            for q in range(k):
                pos = [x0+q*d, y + q*1e-8] # avoid simultaneous collisions
                b = ball(pos, [0,0], R, idx)
                self.balls.append(b)
                idx = idx + 1
            y += np.sqrt(3/4) * d
        # solutions
        self.solutionTime = np.zeros((0))
        self.solutionPos = np.zeros((len(self.balls), 2, 0))
        
    def getFirstBallBallCollision(self):
        n = len(self.balls)
        zv = np.array([0.0, 0.0])
        Tmin = np.Inf
        kMin , lMin = 0, 0
        v1Min , v2Min = zv, zv
        for k in range(n):
            for l in range(k):
                (t, v1, v2) = ballBallCollison(self.balls[k], self.balls[l], self.gravityOn)
                #print("{l,k}", l," ", k, "  ", t)
                if t < Tmin:
                    Tmin = t
                    kMin, lMin = k, l
                    v1Min, v2Min = v1, v2
        return (Tmin, kMin, lMin, v1Min, v2Min)
    
    def getFirstBallBandCollision(self):
        n = len(self.balls)
        vMin = np.array([0.0, 0.0])
        Tmin = np.Inf
        kMin = 0
        for k in range(n):
            (t, v) = ballBandCollision(self.balls[k], self.gravityOn)
            #print("{k}", k,"  ", t)
            if t < Tmin:
                Tmin = t
                vMin = v
                kMin = k
        return (Tmin, kMin, vMin)
                    
    def goToNextCollision(self):
        print("go to next collsion")
        n = len(self.balls)
        
        # get first collision assuming no gravity
        (Tminbb, kbb, lbb, v1Minbb, v2Minbb) = self.getFirstBallBallCollision()
        (Tminb, kb, vMinb) = self.getFirstBallBandCollision()
        Tmin = min(Tminbb, Tminb)
        
        if Tminbb < Tminb:
            print("Tmin = %f (ball-ball) (%d,%d)" % (Tmin, kbb, lbb))
        else:
            print("Tmin = %f (band) (%d)" % (Tmin, kb))
        
        if Tmin == np.Inf:
            raise Exception("No collision, should not happen") 
        
        # Propagate all solutions            
        N = 20
        tEval = np.linspace(0.0, Tmin, N)
        newPos = np.zeros((len(self.balls), 2, N))
        for k in range(n):
            t, posk, spdk = self.balls[k].propagate(tEval, self.gravityOn)
            newPos[k,:,:] = posk
            
        if len(self.solutionTime) == 0:
            self.solutionTime =  tEval
            self.solutionPos = newPos
        else:
            self.solutionTime = np.append(self.solutionTime, tEval[1:] + self.solutionTime[-1], axis = 0)
            self.solutionPos = np.append(self.solutionPos, newPos[:,:,1:], axis = 2) 
            
        # apply collision reflection
        if Tminbb < Tminb:
            self.balls[kbb].v = v1Minbb
            self.balls[lbb].v = v2Minbb
        else:
            self.balls[kb].v = vMinb
            
    def run(self, Ttot, frameRate = 25):
        self.goToNextCollision()     

        while self.solutionTime[-1] < Ttot:
            self.goToNextCollision()

        tMax = self.solutionTime[-1]
        nFrames = int(np.floor(tMax * frameRate))
        self.framesTime = np.array(range(nFrames)) / frameRate
        self.framesPos = np.zeros((len(self.balls), 2, nFrames))
        for n in range(len(self.balls)):
            for k in range(2):
                self.framesPos[n,k,:] = np.interp(self.framesTime, self.solutionTime, self.solutionPos[n,k,:])
                    

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

class animBillard:
    def __init__(self, billard):
        self.billard = billard
        self.W = W
        self.H = H
        self.R = R
        self.nBalls = len(billard.balls)
        self.pos = np.zeros((2, self.nBalls))
        self.spd = np.zeros((2, self.nBalls))
        for k in range(self.nBalls):
            self.pos[:,k] = billard.balls[k].p0
            self.spd[:,k] = billard.balls[k].v
        self.fig, self.ax = plt.subplots()
        self.frames = range(len(billard.solutionTime))
    
    def initAnim(self):
        self.ax.clear()
        self.ln, = self.ax.plot([], [])
        self.ax.set_aspect(aspect='equal', adjustable='box')
        margin = 4 * self.R
        self.ax.set_xlim(left=-self.W - margin, right=self.W + margin)
        self.ax.set_ylim(bottom=-self.H - margin, top=self.H + margin)
        self.ax.grid(b=True)
        self.table = plt.Rectangle([-self.W-self.R,-self.H-self.R], 2*(self.W+self.R), 2*(self.H+self.R), color=[.2,.9,.2])
        self.ax.add_patch(self.table)
        self.ballsCirc = []
        self.spdVec = []
        cmap = get_cmap(2*self.nBalls)
        for k in range(self.nBalls):
            self.ballsCirc.append(plt.Circle(self.pos[:,k], radius=self.R, color=cmap(k)))
            self.ax.add_patch(self.ballsCirc[-1])
        # for k in range(self.nBalls):
        #     rho = 0.1
        #     self.spdVec.append(plt.Arrow(self.pos[0,k],self.pos[1,k], rho*self.spd[0,k], rho*self.spd[1,k], width=.05, color=[.8,.2,.2]))
        #     self.ax.add_patch(self.spdVec[-1])
            
    def update(self, frame):
        for k in range(len(self.ballsCirc)):
            self.ballsCirc[k].set_center(self.billard.framesPos[k,:,frame])
            
    def anim(self):
        return FuncAnimation(self.fig, self.update, self.frames, init_func=self.initAnim, blit=False, repeat_delay=0, interval=0)

            
b = billard(gravityOn = False)
b.run(3.0)
ab = animBillard(b)
ab.initAnim()
an = ab.anim()          