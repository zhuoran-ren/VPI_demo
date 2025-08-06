import numpy as np
from numba import cuda, float32
import math
import time

# Define discretized grid values
X_REL_GRID = np.linspace(0, 5, 21).astype(np.float32)
Y_REL_GRID = np.linspace(-5, 0, 21).astype(np.float32)
VEL_P_GRID = np.linspace(0, 1, 11).astype(np.float32)
VEL_V_GRID = np.linspace(0, 1, 11).astype(np.float32)
ACT_P_GRID = np.linspace(-0.5, 0.5, 11).astype(np.float32)
ACT_V_GRID = np.linspace(-0.5, 0.5, 11).astype(np.float32)

# Planning parameters
HORIZON = 20
BETA = 1.0
dt = 0.1
VEL_P_MAX = 1
VEL_V_MAX = 2
GAMMA = 0.6

# Custom searchsorted implementation for CUDA
@cuda.jit(device=True)
def searchsorted(arr, val):
    for i in range(arr.size):
        if arr[i] >= val:
            return i
    return arr.size - 1

# Compute PET: predictive encounter time
@cuda.jit(device=True)
def compute_pet(x_rel, y_rel, v_p, v_v):
    t_vehicle = x_rel / v_v if v_v > 0 else 1e9
    t_pedestrian = abs(y_rel) / v_p if v_p > 0 else 1e9
    return abs(t_pedestrian - t_vehicle)

# Discrete gamma mapping based on PET value
@cuda.jit(device=True)
def compute_gamma(pet):
    if pet <= 1.770:
        return 3
    elif pet <= 4.962:
        return 2
    else:
        return 1

@cuda.jit(device=True)
def compute_reward(x_rel, y_rel, v_p, v_v, a_p, a_v):
    pet = compute_pet(x_rel, y_rel, v_p, v_v)
    gamma_val = compute_gamma(pet)

    safety_p = -gamma_val * math.exp(v_p)
    safety_v = -gamma_val * math.exp(v_v)

    comfort_p = - a_p ** 2
    comfort_v = - a_v ** 2

    d_offset = 1.0
    target_p = math.exp(1 / (abs(y_rel) + d_offset)) * 10
    target_v = math.exp(1 / (abs(x_rel) + d_offset)) * 10

    r_p = safety_p + comfort_p + target_p
    r_v = safety_v + comfort_v + target_v

    return r_p, r_v

@cuda.jit(device=True)
def apply_transition(x_rel, y_rel, v_p, v_v, a_p, a_v):
    v_p_next = min(max(v_p + a_p * dt, 0.0), VEL_P_MAX)
    v_v_next = min(max(v_v + a_v * dt, 0.0), VEL_V_MAX)
    x_rel_next = x_rel - v_v * dt
    y_rel_next = y_rel + v_p * dt
    return x_rel_next, y_rel_next, v_p_next, v_v_next

@cuda.jit(device=True)
def discretize_state(x, y, v_p, v_v,
                     X_REL_GRID, Y_REL_GRID, VEL_P_GRID, VEL_V_GRID):
    di_x = searchsorted(X_REL_GRID, x)
    di_y = searchsorted(Y_REL_GRID, y)
    di_p = searchsorted(VEL_P_GRID, v_p)
    di_v = searchsorted(VEL_V_GRID, v_v)
    return di_x, di_y, di_p, di_v

@cuda.jit
def compute_value_kernel(V_next, V_out,
                         X_REL_GRID, Y_REL_GRID, VEL_P_GRID, VEL_V_GRID,
                         ACT_P_GRID, ACT_V_GRID):
    """
    GPU kernel to compute the strategic value function for one time step using backward dynamic programming.

    Parameters:
    - V_next: [nx, ny, np, nv] float32, value function at next time step
    - V_out: [nx, ny, np, nv] float32, output value function at current step
    """
    idx = cuda.grid(1)
    nx, ny, np_, nv = X_REL_GRID.size, Y_REL_GRID.size, VEL_P_GRID.size, VEL_V_GRID.size
    total_states = nx * ny * np_ * nv

    if idx >= total_states:
        return

    # Unpack 4D index from flat idx
    i_x = idx // (ny * np_ * nv)
    i_y = (idx // (np_ * nv)) % ny
    i_p = (idx // nv) % np_
    i_v = idx % nv

    x_rel = X_REL_GRID[i_x]
    y_rel = Y_REL_GRID[i_y]
    v_p = VEL_P_GRID[i_p]
    v_v = VEL_V_GRID[i_v]

    best_q = -1e10

    for i_av in range(ACT_V_GRID.size):
        a_v = ACT_V_GRID[i_av]
        q_p_list = cuda.local.array(121, dtype=float32)
        q_v_list = cuda.local.array(121, dtype=float32)
        count = 0

        for i_ap in range(ACT_P_GRID.size):
            a_p = ACT_P_GRID[i_ap]

            # --- Step 1: reward calculation ---
            r_p, r_v = compute_reward(x_rel, y_rel, v_p, v_v, a_p, a_v)

            # --- Step 2: apply dynamics ---
            x_rel_next, y_rel_next, v_p_next, v_v_next = apply_transition(
                x_rel, y_rel, v_p, v_v, a_p, a_v
            )

            # --- Step 3: discretize next state ---
            di_x, di_y, di_p, di_v = discretize_state(
                x_rel_next, y_rel_next, v_p_next, v_v_next,
                X_REL_GRID, Y_REL_GRID, VEL_P_GRID, VEL_V_GRID
            )

            v_next = V_next[di_x, di_y, di_p, di_v]

            q_p = r_p + GAMMA * v_next
            q_v = r_v + GAMMA * v_next

            q_p_list[count] = q_p
            q_v_list[count] = q_v
            count += 1

        # Softmax over Q_p
        max_qp = -1e10
        for i in range(count):
            if q_p_list[i] > max_qp:
                max_qp = q_p_list[i]

        sum_exp = 0.0
        for i in range(count):
            q_p_list[i] = math.exp((q_p_list[i] - max_qp) * BETA)
            sum_exp += q_p_list[i]

        expected_q = 0.0
        for i in range(count):
            prob = q_p_list[i] / sum_exp
            expected_q += prob * q_v_list[i]

        if expected_q > best_q:
            best_q = expected_q

    # Write best Q to output table
    V_out[i_x, i_y, i_p, i_v] = best_q

# Save value table along with grid info
def save_lookup_table(filename, value_table):
    with open(filename, "wb") as f:
        #Write grid sizes
        for grid in [X_REL_GRID, Y_REL_GRID, VEL_P_GRID, VEL_V_GRID]:
            f.write(np.int32(len(grid)).tobytes())

        # Write grid values
        for grid in [X_REL_GRID, Y_REL_GRID, VEL_P_GRID, VEL_V_GRID]:
            f.write(grid.astype(np.float32).tobytes())

        # Write flattened value table
        f.write(value_table.astype(np.float32).flatten().tobytes())
        
def main():
    # Initializa value table
    shape = (len(X_REL_GRID), len(Y_REL_GRID), len(VEL_P_GRID), len(VEL_V_GRID))
    V_gpu = np.zeros(shape, dtype=np.float32)

    # Transfer static grids to device once (not per step)
    d_X_REL = cuda.to_device(X_REL_GRID)
    d_Y_REL = cuda.to_device(Y_REL_GRID)
    d_VEL_P = cuda.to_device(VEL_P_GRID)
    d_VEL_V = cuda.to_device(VEL_V_GRID)
    d_ACT_P = cuda.to_device(ACT_P_GRID)
    d_ACT_V = cuda.to_device(ACT_V_GRID)

    # Precompute kernel configuration
    threads = 128
    total_states = np.prod(shape)
    blocks = (total_states + threads - 1) // threads

    start_time = time.time()
    # Backward dynamic programming loop
    for t in reversed(range(HORIZON)):
        V_next = V_gpu.copy()
        d_V_next = cuda.to_device(V_next)
        d_V_out = cuda.to_device(V_gpu)

        # Launch GPU kernel
        compute_value_kernel[blocks, threads](
            d_V_next, d_V_out, 
            d_X_REL, d_Y_REL, d_VEL_P, d_VEL_V,
            d_ACT_P, d_ACT_V
        )

        V_gpu = d_V_out.copy_to_host()
        print(f"Step {t} completed.")

    # Save the Value table
    filename = "data/Strategic_value_table.bin"
    save_lookup_table(filename, V_gpu)
    print("Saved value table to 'Strategic_value_table.bin'")
    
    end_time = time.time()
    print(f"Total running time for one loop: {end_time - start_time:.4f} s")

if __name__ == "__main__":
    main()