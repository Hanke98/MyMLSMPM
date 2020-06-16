import taichi as ti
import random
import numpy as np
random.seed(10)

ti.init(arch=ti.metal)

dim = 2
width = 600
height = 600 

n_particles = 8199
n_grid      = 128
dx          = 1 / n_grid
inv_dx      = 1 / dx
dt          = 2.0e-4 / 10
p_vol       = (dx * 0.5)**2
p_rho       = 1
p_mass      = p_vol * p_rho

E, nu       = 0.1e4, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters

x = ti.Vector(dim, dt=ti.f32, shape=n_particles)
v = ti.Vector(dim, dt=ti.f32, shape=n_particles)
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid)) # grid node mass
F = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)
C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)
J = ti.var(dt=ti.f32, shape=n_particles) # plastic deformation
material = ti.var(dt=ti.i32, shape=n_particles) # material id

def init():
    for i in range(n_particles):
        x[i] = [random.random() * 0.2 + 0.2, random.random() * 0.2 + 0.2]

init()

@ti.kernel
def step():

    for i, j in grid_v:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    # 1. p2g
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # 1.1 F update
        F[p] = (ti.Matrix.identity(dt=ti.f32, n=2) + dt * C[p]) @ F[p]
        r, _ = ti.polar_decompose(F[p])
        U, sig, V = ti.svd(F[p])
        J[p] = F[p].determinant()
        # 1.2 Cauchy Stress
        mu = 0
        PF = 2 * mu * (F[p] - U @ V.T()) @ F[p].T() + lambda_0 * (J[p] - 1) * J[p] * ti.Matrix.identity(dt=ti.f32, n=2)
        Dinv = 4 * inv_dx * inv_dx
        stress = -dt * p_vol * Dinv * PF
        # 1.3 affine
        affine = stress + p_mass * C[p]
        
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    # 2. update grad momentum
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]  # Momentum to velocity
            grid_v[i, j][1] -= dt * 200  # gravity
            if i < 3 and grid_v[i, j][0] < 0:          grid_v[i, j][0] = 0 # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:          grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0
   
    # 3. g2p
    for p in x: # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)): # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p] 


group_size = n_particles // 3

@ti.kernel
def initialize():
    for i in range(n_particles):
        x[i] = [
            ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
            ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)
        ]
        v[i] = ti.Matrix([0, 0])
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        C[i] = ti.Matrix([[0, 0], [0, 0]])
        J[i] = 1
        material[i] = i // group_size  
initialize()

gui = ti.GUI("MyMLSMPM", (width, height))
for i in range(1000000):
    gui.clear(0x112F41)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
    gui.circles(x.to_numpy(), radius=1.5, color=colors[material.to_numpy()])
    # gui.circles(x.to_numpy(), color=colors[material.to_numpy()], radius=1.50)
    for _ in range(10):
        step()
    gui.show()
