#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 15:30:08 2021

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
# assuming unit masses and gravitational constant
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
    z0 = np.array([0,0,0,0]);
    fInt = lambda tt, zz : np.concatenate((zz[2:4], force(p0, v0, c, tt)))
    tSpan = [0, T]
    ivpSol = integrate.solve_ivp(fInt, tSpan, z0, max_step = 0.01)
    # return (time, pos, speed)
    return (ivpSol.t, ivpSol.y[0:2,:], ivpSol.y[2:4,:])
    
    

p0 = np.asarray([1.0,1.0])
v0 = np.asarray([0.0,1.0])
c = np.asarray([0.0,0.0])
(t, dpos, dspd) = simulate(p0, v0, c,1)

pos = dpos + np.asarray([t * v0[0] + p0[0], t * v0[1] + p0[1]])

plt.plot(pos[0,:], pos[1,:])
plt.axis('equal')


class CollisionType(Enum):
    EARLIEST = 0 # look for smallest positive collision time
    CLOSEST = 1 # look for smallest absolute collision time

# collison time and positions between two balls
def ballBallCollison(pa, va, pb, vb, R, collisionType):
    pa = np.asarray(pa)
    pb = np.asarray(pb)
    va = np.asarray(va)
    vb = np.asarray(vb)
    d = pa - pb
    w = va - vb
        
    A = np.dot(w, w)
    B = 2 * np.dot(d, w)
    C = np.dot(d, d) - 4 * R**2
    discr = B**2 - 4 * A * C
    
    if (A == 0.0):
        return (np.Inf, pa, pb)
    
    t1, t2 = 0, 0
    if discr < 0:
        return (np.Inf, pa, pb)
    else:
        t1 = (-B + np.sqrt(discr)) / (2 * A)
        t2 = (-B - np.sqrt(discr)) / (2 * A)
    
    t = 0
    if collisionType == CollisionType.EARLIEST:
        if max(t1, t2) < 0.0:
            return (np.Inf, pa, pb)
        else:
            t = min(t1, t2) if min(t1, t2) >= 0.0 else max(t1, t2)
    
    elif collisionType == CollisionType.CLOSEST:
        t = t1 if abs(t1) < abs(t2) else t2
        
    else:
        raise Exception("Undefined collision type") 
        
    paColl = pa + t * va
    pbColl = pb + t * vb
    return (t, paColl, pbColl)
    
ballBallCollison([0,1], [1,-1], [0,-1], [1,1], 0.1, CollisionType.EARLIEST)


class CollisionBand(Enum):
    RIGHT = 0
    TOP = 1
    LEFT = 2
    BOTTOM = 4
    NONE = 5
    
# collision time and position between ball and band on 2Wx2H billard centered at (0,0)
def ballBandCollision(p, v, W, H, collisionType, ignoreBand = CollisionBand.NONE):
    p = np.asarray(p)
    v = np.asarray(v)
    
    if (np.linalg.norm(v) == 0.0):
        return (np.Inf, p, [0,1], CollisionBand.NONE)
    
    # Find intersection with edge defined by position q and direction d
    def intersection(q, d):
        q = np.asarray(q)
        d = np.asarray(d)
        A = np.array([v, -d]).transpose()
        if (np.linalg.det(A) == 0.0):
            return (np.array([np.nan, np.nan]), np.Inf)
        b = q - p
        x = np.linalg.solve(A, b)
        tSol = x[0]
        pSol = p + v * tSol
        return (pSol, tSol)
    
    pRIGHT,  tRIGHT  = intersection([ W,  0], [0,1])
    pTOP,    tTOP    = intersection([ 0,  H], [1,0])
    pLEFT,   tLEFT   = intersection([-W,  0], [0,1])
    pBOTTOM, tBOTTOM = intersection([ 0, -H], [1,0])
    
    if ignoreBand == CollisionBand.RIGHT:
        tRIGHT = np.Inf
    elif ignoreBand == CollisionBand.TOP:
        tTOP = np.Inf
    elif ignoreBand == CollisionBand.LEFT:
        tLEFT = np.Inf
    elif ignoreBand == CollisionBand.BOTTOM:
        tBOTTOM = np.Inf
    
    tList = np.array([tRIGHT, tTOP, tLEFT, tBOTTOM])
    typeList = [CollisionBand.RIGHT, CollisionBand.TOP, CollisionBand.LEFT, CollisionBand.BOTTOM]
    
    indMin = -1
    t = 0
    band = CollisionBand.NONE
    if collisionType == CollisionType.EARLIEST:       
        tList[tList < 0] = np.Inf
        indMin = np.where(tList == np.amin(tList))[0][0]
    else: # CLOSEST
        tListAbs = np.abs(tList)
        indMin = np.where(tListAbs == np.amin(tListAbs))[0][0]
        
    t = tList[indMin]
    band = typeList[indMin]
        
    if (t < np.Inf):
        pColl = p + v * t
    else:
        pColl = p
            
    if ((indMin == 0) or (indMin == 2)):
        return (t, pColl, [0,1], band)
    else:
        return (t, pColl, [1,0], band)
    
       
    
p = np.array([0.1, 0.1])
v = np.array([0.8, 0.1])
W = 0.4
H = 0.7
ballBandCollision(p, v, W, H, CollisionType.EARLIEST)    


def speedReflexionBand(v, ax):
    v = np.asarray(v)
    ax = np.asarray(ax)
    ax = ax / np.linalg.norm(ax)
    n = np.array([ax[1], -ax[0]])
    A1 = np.array([ax, n]).transpose()
    A2 = np.array([ax, -n])
    return np.matmul(A1, np.matmul(A2,v))

def speedReflexionBallBall(va, vb, ax):
    va = np.asarray(va)
    vb = np.asarray(vb)
    ax = np.asarray(ax)
    ax = ax / np.linalg.norm(ax)
    n = np.array([ax[1], -ax[0]])
    A1 = np.array([ax, n]).transpose()
    
    va1 = np.matmul(A1, va)
    vb1 = np.matmul(A1, vb)
    
    tmp = va1[1]
    va1[1] = vb1[1]
    vb1[1] = tmp
    
    A2 = np.array([ax, n])
    return (np.matmul(A2, va1), np.matmul(A2, vb1))


class ball:
    def __init__(self, p0 = [0,0], v = [0,0], R = 60/1000):
        self.p0 = np.asarray(p0)
        self.v = np.asarray(v)
        self.R = R


class CollisionCase(Enum):
    BALLBALL = 0
    BALLBAND = 1
    NONE = 2

    
class billard:
    def __init__(self, W = 1.3/2, H = 2.5/2, c = [1.5, 0.0], M = 80.0, G = 0*6.67e-11, R = 60/1000/2):
        self.W = W
        self.H = H
        self.c = np.asarray(c) # player mass position
        self.M = M # player mass
        self.G = G
        self.R = R
        # balls
        self.balls = [ball([4*R,-H/2], [1,2], R)] # white ball
        self.lastCollidingBallsInd = {}
        self.lastBandHit = CollisionBand.NONE
        self.lastCollisionCase = CollisionCase.NONE
        n = 1 # 3
        d = 1.1 * 2 * R
        y = 0
        for k in range(1,n+1):
            x0 = -d * (k - 1) / 2
            for q in range(k):
                pos = [x0+q*d, y + q*1e-8] # avoid simultaneous collisions
                b = ball(pos, [0,0], R)
                self.balls.append(b)
            y += np.sqrt(3/4) * d
                
    def ballBallCollison(self, ba, bb):
        A =  np.array([[0,1],[-1,0]])
        # Collision between balls with gravity
        eps = self.G * self.M
        collNoGrav = ballBallCollison(ba.p0, ba.v, bb.p0, bb.v, self.R, CollisionType.EARLIEST) # returns (t, paColl, pbColl)
        if collNoGrav[0] == np.Inf:
            return (np.Inf, copy(ba), copy(bb))
        else:
            if eps == 0.0: # no gravity effect
                T = collNoGrav[0] 
                baEnd = copy(ba)
                bbEnd = copy(bb)
                baEnd.p0 = collNoGrav[1]
                bbEnd.p0 = collNoGrav[2]
                baEnd.v, bbEnd.v = speedReflexionBallBall(baEnd.v, bbEnd.v, np.matmul(A, bbEnd.p0 - baEnd.p0))
                return (T, baEnd, bbEnd)
            else:
                (T, pa, pb) = collNoGrav # collision approximated without gravity
                _, dpa, dva = simulate(ba.p0, ba.v, self.c, T) # returns (time, pos, speed)
                _, dpb, dvb = simulate(bb.p0, bb.v, self.c, T) # returns (time, pos, speed)
                pa += eps * dpa[:,-1] # corrected position a
                pb += eps * dpb[:,-1] # corrected position b
                va = ba.v + eps * dva[:,-1] # corrected speed a
                vb = bb.v + eps * dvb[:,-1] # corrected speed b
                TCorr, pa, pb = ballBallCollison(pa, pb, va, vb, self.R, CollisionType.CLOSEST) # collision correction
                if TCorr == np.Inf:
                    return (np.Inf, copy(ba), copy(bb))
                T += TCorr
                baEnd = copy(ba)
                bbEnd = copy(bb)
                baEnd.p0 = pa
                bbEnd.p0 = pb
                baEnd.v = va
                bbEnd.v = vb
                baEnd.v, bbEnd.v = speedReflexionBallBall(baEnd.v, bbEnd.v, np.matmul(A, bbEnd.p0 - baEnd.p0))
                return (T, baEnd, bbEnd)
            
    def ballBandCollision(self, b):
        # Collision against band with gravity
        eps = self.G * self.M
        T, pColl, ax, band = ballBandCollision(b.p0, b.v, self.W, self.H, CollisionType.EARLIEST, self.lastBandHit)
        if T == np.Inf:
            return (np.Inf, copy(b), CollisionBand.NONE)
        _, dp, dv = simulate(b.p0, b.v, self.c, T) # returns (time, pos, speed)
        p = pColl + eps * dp[:,-1] # corrected position a
        v = b.v + eps * dv[:,-1]
        
        if eps > 0:
            TCorr, pColl, ax, _ = ballBandCollision(p, v, self.W, self.H, CollisionType.CLOSEST, CollisionBand.NONE)
            T += TCorr
            
        bEnd = copy(b)
        bEnd.p0 = pColl
        bEnd.v = speedReflexionBand(b.v, ax)
        
        return (T, bEnd, band)
    
    def getFirstBallBallCollision(self):
        n = len(self.balls)
        Tmatrix = np.zeros((n,n)) + np.Inf
        for k in range(n):
            for l in range(k):
                if {k,l} != self.lastCollidingBallsInd:
                    Tmatrix[k,l],_ ,_ = self.ballBallCollison(self.balls[k], self.balls[l])
        ind = np.where(Tmatrix == np.amin(Tmatrix))
        k = ind[0][0]
        l = ind[1][0]
        return (Tmatrix[k,l], l, k)
    
    def getFirstBallBandCollision(self):
        n = len(self.balls)
        Tlist = np.zeros((n)) + np.Inf
        bandList = []
        for k in set(range(n)).difference(self.lastCollidingBallsInd):
            Tlist[k], _, band = self.ballBandCollision(self.balls[k])
            bandList.append(band)
        ind = np.where(Tlist == np.amin(Tlist))
        k = ind[0][0]
        return (Tlist[k], k, bandList[k])
    
    def goToNextBallsState(self):
        (Tbb, lbb, kbb) = self.getFirstBallBallCollision()
        (Tb, kb, band) = self.getFirstBallBandCollision()
        Tmin = min(Tbb, Tb)
        # Propagate solutions before updating state
        # TODO !!
        # Update state
        if Tbb < Tb:
            T , ba, bb = self.ballBallCollison(self.balls[kbb], self.balls[lbb])
            self.balls[kbb].p0 = ba.p0
            self.balls[kbb].v = ba.v
            self.balls[lbb].p0 = bb.p0
            self.balls[lbb].v = bb.v
            self.lastCollidingBallsInd = {kbb, lbb}
            self.lastCollisionCase = CollisionCase.BALLBALL
            self.lastBandHit = CollisionBand.NONE
            print("Collision time : ",T, "(ball-ball)")
        else:
            T, b, band = self.ballBandCollision(self.balls[kb])
            self.balls[kb].p0 = b.p0
            self.balls[kb].v = b.v
            self.lastCollidingBallsInd = {kb}
            self.lastCollisionCase = CollisionCase.BALLBAND
            self.lastBandHit = band
            print("Collision time : ",T,  "(ball-band)")


    
b = billard()

b.getFirstBallBallCollision()
b.getFirstBallBandCollision()

class animBillard:
    def __init__(self, billard):
        self.billard = billard
        self.W = billard.W
        self.H = billard.H
        self.R = billard.R
        self.nBalls = len(billard.balls)
        self.pos = np.zeros((2, self.nBalls))
        self.spd = np.zeros((2, self.nBalls))
        for k in range(self.nBalls):
            self.pos[:,k] = billard.balls[k].p0
            self.spd[:,k] = billard.balls[k].v
        self.fig, self.ax = plt.subplots()
    
    def initAnim(self):
        self.ax.clear()
        self.ln, = self.ax.plot([], [])
        self.ax.set_aspect(aspect='equal', adjustable='box')
        margin = 4 * self.R
        self.ax.set_xlim(left=-self.W - margin, right=self.W + margin)
        self.ax.set_ylim(bottom=-self.H - margin, top=self.H + margin)
        #self.ax.set_xbound(lower=-1.1*self.W, upper=1.1*self.W)
        #self.ax.set_ybound(lower=-1.1*self.H, upper=1.1*self.H)
        self.ax.grid(b=True)
        self.table = plt.Rectangle([-self.W-self.R,-self.H-self.R], 2*(self.W+self.R), 2*(self.H+self.R), color=[.2,.9,.2])
        self.ax.add_patch(self.table)
        self.ballsCirc = []
        self.spdVec = []
        for k in range(self.nBalls):
            self.ballsCirc.append(plt.Circle(self.pos[:,k], radius=self.R))
            self.ax.add_patch(self.ballsCirc[-1])
        for k in range(self.nBalls):
            rho = 0.1
            self.spdVec.append(plt.Arrow(self.pos[0,k],self.pos[1,k], rho*self.spd[0,k], rho*self.spd[1,k], width=.05, color=[.8,.2,.2]))
            self.ax.add_patch(self.spdVec[-1])
            
b2 = billard()
ab = animBillard(b2)
ab.initAnim()

b2.goToNextBallsState()
ab2 = animBillard(b2)
ab2.initAnim()

b2.goToNextBallsState()
ab3 = animBillard(b2)
ab3.initAnim()

b2.goToNextBallsState()
ab4 = animBillard(b2)
ab4.initAnim()

b2.goToNextBallsState()
ab5 = animBillard(b2)
ab5.initAnim()



