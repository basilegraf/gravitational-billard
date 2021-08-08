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
from copy import copy

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
    
    t1, t2 = 0, 0
    if discr < 0:
        return (np.Inf,)
    else:
        t1 = (-B + np.sqrt(discr)) / (2 * A)
        t2 = (-B - np.sqrt(discr)) / (2 * A)
    
    t = 0
    if collisionType == CollisionType.EARLIEST:
        if max(t1, t2) < 0.0:
            return (np.Inf,)
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
    
# collision time and poition between ball and band on 2Wx2H billard centered at (0,0)
def ballBandCollision(p, v, W, H):
    p = np.asarray(p)
    v = np.asarray(v)
    band = CollisionBand.NONE
    t = np.Inf
    
    if abs(p[0]) > W or abs(p[1]) > H:
        return (np.Inf, band)
    
    if v[0] >= 0:
        tRight = (W - p[0]) / v[0]
        if tRight < t:
            band = CollisionBand.RIGHT
            t = tRight
    else:
        tLeft = (-W - p[0]) / v[0]
        if tLeft < t:
            band = CollisionBand.LEFT
            t = tLeft
        
    if v[1] >= 0:
        tTop = (H - p[1]) / v[1]
        if tTop < t:
            band = CollisionBand.TOP
            t = tTop
    else:
        tBottom = (-H - p[1]) / v[1]
        if tBottom < t:
            band = CollisionBand.BOTTOM
            t = tBottom
            
    pColl = p + v * t
    return (t, pColl, band)
        
    
p = np.array([0.1, 0.1])
v = np.array([0.8, 0.1])
W = 0.4
H = 0.7
ballBandCollision(p, v, W, H)    


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
    n = np.array([ax[1], -ax[0]])
    A1 = np.array([ax, n]).transpose()
    
    va1 = np.matmul(A1, va)
    vb1 = np.matmul(A1, vb)
    
    tmp = va1[1]
    va1[1] = vb1[1]
    vb1[1] = tmp
    
    A2 = np.array([ax, -n])
    return (np.matmul(A2, va1), np.matmul(A2, vb1))


class ball:
    def __init__(self, p0 = [0,0], v = [0,0], R = 60/1000):
        self.p0 = np.asarray(p0)
        self.v = np.asarray(v)
        self.R = R

    
    
class billard:
    def __init__(self, W = 1.3/2, H = 2.5/2, c = [1.5, 0.0], M = 80.0, G = 6.67e-11, R = 60/1000):
        self.W = W
        self.H = H
        self.c = np.asarray(c) # player mass position
        self.M = M # player mass
        self.G = G
        self.R = R
        # balls
        self.balls = [ball([0,-H/2], [0,2], R)] # white ball
        n = 3
        d = 1.01 * R
        y = 0
        for k in range(1,n+1):
            x0 = d * (k - 1) / 2
            for q in range(k):
                pos = [x0+q*d, y + q*1e-8] # avoid simultaneous collisions
                b = ball(pos, [0,0], R)
                self.balls.append(b)
                y += np.sqrt(3/4) * d
                
    def ballBallCollison(self, ba, bb):
        # Collision between balls with gravity
        eps = self.G * self.M
        collNoGrav = ballBallCollison(ba.p, ba.v, bb.p, bb.v, self.R, CollisionType.EARLIEST) # returns (t, paColl, pbColl)
        if collNoGrav[0] == np.Inf:
            return (np.Inf, copy(ba), copy(bb))
        else:
            if eps == 0.0: # no gravity effect
                T = collNoGrav[0] 
                baEnd = copy(ba)
                bbEnd = copy(bb)
                baEnd.p0 = collNoGrav[1]
                bbEnd.p0 = collNoGrav[2]
                baEnd.v, bbEnd.v = speedReflexionBallBall(baEnd.v, bbEnd.v, bbEnd.p0 - baEnd.p0)
                return (T, baEnd, bbEnd)
            else:
                (T, pa, pb) = collNoGrav # collision approximated without gravity
                _, dpa, dva = simulate(ba.p, ba.v, self.c, T) # returns (time, pos, speed)
                _, dpb, dvb = simulate(bb.p, bb.v, self.c, T) # returns (time, pos, speed)
                pa += eps * dpa # corrected position a
                pb += eps * dpb # corrected position b
                va = ba.v + eps * dva # corrected speed a
                vb = bb.v + eps * dvb # corrected speed b
                collGrav = ballBallCollison(pa, pb, va, vb, self.R, CollisionType.CLOSEST) # collision correction
                TCorr, pa, pb = collGrav
                if TCorr == np.Inf:
                    return (np.Inf, copy(ba), copy(bb))
                T += TCorr
                baEnd = copy(ba)
                bbEnd = copy(bb)
                baEnd.p0 = pa
                bbEnd.p0 = pb
                baEnd.v = va
                bbEnd.v = vb
                baEnd.v, bbEnd.v = speedReflexionBallBall(baEnd.v, bbEnd.v, bbEnd.p0 - baEnd.p0)
                return (T, baEnd, bbEnd)

    
    
    
    