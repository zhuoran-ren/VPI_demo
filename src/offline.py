import torch
from scipy.special import softmax

# Get the current device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Strategic planner parameters
X_REL_GRID = torch.linspace(0, 10, 21).to(device)
Y_REL_GRID = torch.linspace(-5, 0, 21).to(device)
VEL_P_GRID = torch.linspace(0, 1, 11).to(device)
VEL_V_GRID = torch.linspace(0, 2, 11).to(device)
ACT_P_GRID = torch.linspace(-1, 1, 11).to(device)
ACT_V_GRID = torch.linspace(-1, 1, 11).to(device)

HORIZON = 5
BETA = 1.0
dt = 0.1

VEL_P_MAX = 1
VEL_V_MAX = 2

# states x actions pair: shape [21x21x11x11x11x11, 6]
state_action_grid = torch.cartesian_prod(
    X_REL_GRID, 
    Y_REL_GRID,
    VEL_P_GRID,
    VEL_V_GRID,
    ACT_P_GRID, 
    ACT_V_GRID
).to(device)

print("hahah")
def compute_pet(x_rel, y_rel, v_p, v_v):

    t_vehicle = x_rel / v_v if v_v > 0 else torch.inf
    t_pedestrian = abs(y_rel) / v_p if v_p > 0 else torch.inf

    return abs(t_pedestrian - t_vehicle)

def compute_gamma(pet):
    if pet <= 1.770:
        return 3
    elif pet <= 4.962:
        return 2
    else:
        return 1

# Reward function: combines safety and progress
def reward(state, action):
    x_rel, y_rel, v_p, v_v = state
    a_p, a_v = action
    PET = compute_pet(x_rel, y_rel, v_p, v_v)
    gamma = compute_gamma(PET)

    safety_p = -gamma * torch.exp(v_p)
    safety_v = -gamma * torch.exp(v_v)

    comfort_p = - a_p ** 2
    comfort_v = -a_v ** 2

    d_offset = 1
    target_p = torch.exp(1 / (abs(y_rel) + d_offset))
    target_v = torch.exp(1 / (abs(x_rel) + d_offset))

    reward_vehicle = safety_v + comfort_v + target_v
    reward_pedestrian = safety_p + comfort_p + target_p
    return reward_vehicle, reward_pedestrian

# Simplified vehicle dynamics: return the next state
def transition(state, action):
    x_rel, y_rel, v_p, v_v = state
    a_p, a_v = action

    v_p_next = torch.clamp(v_p + a_p * dt, 0.0, VEL_P_MAX)
    v_v_next = torch.clamp(v_v + a_v * dt, 0.0, VEL_V_MAX)

    x_rel_next = x_rel - v_v * dt
    y_rel_next = y_rel + v_p * dt
    
    return (x_rel_next, y_rel_next, v_p_next, v_v_next)

def discretize(state):
    """
    Discretize a continuous state into the nearest grid index in each dimension.
    """
    x, y, v_p, v_v = state

    dis_x_rel = torch.searchsorted(X_REL_GRID, x, side="left")
    dis_y_rel = torch.searchsorted(Y_REL_GRID, y, side="left")
    dis_v_p = torch.searchsorted(VEL_P_GRID, v_p, side="left")
    dis_v_v = torch.searchsorted(VEL_V_GRID, v_v, side="left")

    return (dis_x_rel, dis_y_rel, dis_v_p, dis_v_v)

def softmax(q_values):
    """
    Compute a softmax distribution over a list of Q-values.
    This is used to model the pedestrian's stochastic response.
    """
    q_values = torch.tensor(q_values)
    q_values -= torch.max(q_values)

    exp_q = torch.exp(BETA * q_values)
    probs = exp_q / torch.sum(exp_q)

    return probs

def compute_q(state_grid, ):
    pass
def compute_strategic_value_cpu(GAMMA=0.6):
    """
    Compute the Stackelberg strategic value function using backward dynamic programming.

    Returns:
    - V: dict mapping (x_rel, y_rel, v_p, v_v) to a float value.
    """
    
    # Initialize V tensor with appropriate size for all states
    V = torch.zeros((len(X_REL_GRID), len(Y_REL_GRID), len(VEL_P_GRID), len(VEL_V_GRID)), device=device)

    # Backward dynamic programming
    for t in reversed(range(HORIZON)):
        print(f"Backward step {t} / {HORIZON}")

        for i_x, x_rel in enumerate(X_REL_GRID):
            for i_y, y_rel in enumerate(Y_REL_GRID):
                for i_p, v_p in enumerate(VEL_P_GRID):
                    for i_v, v_v in enumerate(VEL_V_GRID):

                        state = (x_rel, y_rel, v_p, v_v)
                        best_q = -torch.inf

                        for a_v in ACT_V_GRID:
                            Q_p_list = []
                            Q_v_list = []

                            for a_p in ACT_P_GRID:
                                action = (a_p, a_v)

                                # Get reward and transition
                                r_v, r_p = reward(state, action)
                                next_state = transition(state, action)
                                next_idx = discretize(next_state)

                                # Lookup next state value
                                v_next = V[next_idx]

                                Q_p = r_p + GAMMA * v_next
                                Q_v = r_v + GAMMA * v_next
                                Q_p_list.append(Q_p)
                                Q_v_list.append(Q_v)

                            p_probs = softmax(Q_p_list)

                            expected_q = torch.sum(torch.tensor([p*q for p, q in zip(p_probs, Q_v_list)], device=device))

                            if expected_q > best_q:
                                best_q = expected_q

                        V[i_x, i_y, i_p, i_v] = best_q
    return V

def compute_strategic_value_cuda(GAMMA=0.6):
    """
    Compute the Stackelberg strategic value function using backward dynamic programming.

    Returns:
    - V: dict mapping (x_rel, y_rel, v_p, v_v) to a float value.
    """
    # Initialize V tensor with appropriate size for all states
    V = torch.zeros((len(X_REL_GRID), len(Y_REL_GRID), len(VEL_P_GRID), len(VEL_V_GRID)), device=device)

    # Backward dynamic programming - vectorized approach
    for t in reversed(range(HORIZON)):
        print(f"Backward step {t} / {HORIZON}")

        # Generate all possible actions as a tensor
        actions = torch.cartesian_prod(ACT_P_GRID, ACT_V_GRID).to(device)

        # Prepare tensor for all state combinations
        states = torch.cartesian_prod(X_REL_GRID, Y_REL_GRID, VEL_P_GRID, VEL_V_GRID).to(device)

        # Compute rewards for all states and actions at once
        rewards = torch.stack([reward(state, action) for state in states for action in actions], dim=0).to(device)
        
        # Transition and discretize for all states and actions
        next_states = [transition(state, action) for state in states for action in actions]
        next_idx = [discretize(ns) for ns in next_states]

        # Vectorized lookup in V tensor for next states
        v_next = torch.tensor([V[idx] for idx in next_idx], device=device)

        # Compute Q values for each action pair and state
        Q_p = rewards[:, 0] + GAMMA * v_next
        Q_v = rewards[:, 1] + GAMMA * v_next

        # Compute softmax probabilities
        p_probs = softmax(Q_p)

        # Calculate expected Q value
        expected_q = torch.sum(p_probs * Q_v, dim=0)

        # Reshape expected Q value back into the V tensor for all states
        V[:] = expected_q.view(len(X_REL_GRID), len(Y_REL_GRID), len(VEL_P_GRID), len(VEL_V_GRID))

    return V

def main():
    print("Strating strategoc value computation...")

    # Run backward DP
    V = compute_strategic_value_cuda()

    print("Starting strategic value computation ...")

    print("Strategic value computation complete.")
    print(f"Total number of stored state values: {len(V)}")

    # Example: print some sampled values
    test_state = (5.0, -2.5, 0.5, 1.0)
    value = V.get(test_state, None)
    if value is not None:
        print(f"V({test_state}) = {value:.4f}")
    else:
        print(f"State {test_state} not found in V")

    # Save to disk
    import pickle
    with open("strategic_value.pkl", "wb") as f:
        pickle.dump(V, f)
    print("Value function saved to strategic.value.pkl")


if __name__ == "__main__":
    main()