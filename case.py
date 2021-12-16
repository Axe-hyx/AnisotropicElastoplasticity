
import taichi as ti
import random
import math

ti.init(arch = ti.cpu)

dim = 2
n_grid = 256
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 4.0e-5

n_type2 = 200
Length = 0.75
sl = Length/ (n_type2-1)
volume2 =  dx*Length / (n_type2)

p_rho = 1

start_pos = ti.Vector([0.2, 0.8])

x2 = ti.Vector.field(2, dtype=float, shape=n_type2) # position
v2 = ti.Vector.field(2, dtype=float, shape=n_type2) # velocity
C2 = ti.Matrix.field(2, 2, dtype=float, shape=n_type2) # affine velocity field

grid_v = ti.Vector.field(2, dtype= float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))

@ti.kernel
def Reset():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

@ti.kernel
def Particle_To_Grid():
    for p in x2:
        base = (x2[p] * inv_dx - 0.5).cast(int)
        fx = x2[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2]
        affine = C2[p]
        mass = volume2* p_rho
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i,j])
            weight = w[i][0]*w[j][1]
            grid_m[base + offset] += weight * mass
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass * (v2[p] +  affine@dpos)

bound = 3
@ti.kernel
def Grid_Collision():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= dt * 9.80

            if i < bound and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0
            if i > n_grid - bound and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0
            if j < bound and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0
            if j > n_grid - bound and grid_v[i, j].y > 0:
                grid_v[i, j].y = 0


@ti.kernel
def Grid_To_Particle():
    for p in x2:
        base = (x2[p] * inv_dx - 0.5).cast(int)
        fx = x2[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        new_x = ti.Vector.zero(float, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
            offset = ti.Vector([i, j]).cast(float)
            new_x += weight * dx * (base.cast(float) + offset)
        v2[p] = new_v
        x2[p] += dt * v2[p]
        C2[p] = new_C

@ti.kernel
def initialize():
    for i in range(n_type2):
        x2[i] = ti.Vector([start_pos[0]+ i * sl, start_pos[1]])
        v2[i] = ti.Matrix([0, 0])
        C2[i] =  ti.Matrix([[0,0],[0,0]])

def main():
    initialize()
    gui = ti.GUI("MRE", (512, 512))
    frame = 0
    for __ in range(100):
        maximal = 0.
        for _ in range(100):
            Reset()
            Particle_To_Grid()
            Grid_Collision()
            Grid_To_Particle()
            x2_np = x2.to_numpy()
            for i in range(n_type2 - 1):
                d = x2_np[i+1] - x2_np[i]
                maximal = max(maximal, math.fabs(d[1]))
            frame += 1

        print("maximal " + str(maximal) + "\tduring frame[" + str(frame - 100)+ "," +str(frame) + "]", flush=True)

if __name__ == "__main__":
    main()
