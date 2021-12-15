# script from https://zhuanlan.zhihu.com/p/414356129

import taichi as ti
import random
import math

ti.init(arch = ti.cpu)

# number of lines
N_Line = 12
# line space distance
dlx = 0.009
# type2 particle count per line
ln_type2 = 200

start_pos = ti.Vector([0.2, 0.8])

ln_type3 = ln_type2 - 1
n_type2 = N_Line* ln_type2
n_type3 = N_Line* ln_type3

#line length
Length = 0.75
sl = Length/ (ln_type2-1)

#type2
x2 = ti.Vector.field(2, dtype=float, shape=n_type2) # position

#type3
D3_inv = ti.Matrix.field(2, 2, dtype=float, shape=n_type3)
d3 = ti.Matrix.field(2, 2, dtype=float, shape=n_type3)

n_segment = n_type3

ROT90 = ti.Matrix([[0,-1.0],[1.0,0]])

#get type2 from type3
@ti.func
def GetType2FromType3(index):
    index += index // ln_type3
    return index, index+1

@ti.kernel
def initialize():
    for i in range(n_type2):
        sq = i // ln_type2
        x2[i] = ti.Vector([start_pos[0]+ (i- sq* ln_type2) * sl, start_pos[1] + sq* dlx])

    for i in range(n_segment):
        l, n = GetType2FromType3(i)

        # error here, dp0 is not guaranteed the same for every point
        dp0 = x2[n] - x2[l]
        dp1 = ROT90@dp0
        dp1 /= dp1.norm()
        d3[i] = ti.Matrix.cols([dp0,dp1])
        D3_inv[i] = d3[i].inverse()

def main():
    initialize()
    frame = 1

    f = open("results/iii" + str(frame), "w")
    d3_ny = d3.to_numpy()
    D3_inv_ny = D3_inv.to_numpy()
    for i in range(n_type3):
        f.write(str(i) + "\nD3_inv\n" + str(D3_inv_ny[i]) + "\nd3\n" + str(d3_ny[i]) + "\n")

    f.close()

if __name__ == "__main__":
    main()
