import numpy as np
import nashpy as nash
import random

class ChickenGame:
    def __init__(self):
        # Constant parameters
        self.theta_1 = 0.946
        self.theta_2 = 0.975
        self.theta_3 = 0.981
        self.theta_4 = 0.567
        self.V_s = 0.750

        # Physical sizes
        self.vehicle_length = 0.8
        self.vehicle_width = 0.4
        self.ped_radius = 0.2

        # Velocity limits
        self.vv_max = 10
        self.vp_max = 10

        # Initial positions
        self.p_x = 5.0
        # self.p_y = np.random.uniform(0.0, 2.0)
        self.p_y = 1.5
        self.v_x = 3.0
        self.v_y = 3.0

        self.vp_y = random.uniform(0, self.vp_max)
        self.vv_x = random.uniform(0, self.vv_max)

        # Time
        self.tp = 0.0
        self.tv = 0.0
        self.dt = 0.01

        # Conflict region size (square zone)
        self.conflict_x = self.p_x
        self.conflict_y = self.v_y

        # Tresholds for state dertermination
        self.tresh_v = self.ped_radius + self.vehicle_length / 2
        self.tresh_p = self.ped_radius + self.vehicle_width / 2

        # Initial State
        self.state = ("before", "before")

        # Historical data for visualization
        self.v_pass_prob_history = []

    def get_zone(self, distance, tresh):
        """Return 'before', 'in', or 'after' based on location."""
        if distance > tresh:
            return "before"
        elif distance <  - tresh:
            return "after"
        else:
            return "in"

    def determine_state(self):
        """Determine system state based on relative positions."""
        veh_distance = self.conflict_x - self.v_x
        ped_distance = self.conflict_y - self.p_y

        veh_zone = self.get_zone(veh_distance, self.tresh_v)
        ped_zone = self.get_zone(ped_distance, self.tresh_p)

        self.state = (veh_zone, ped_zone)

        if self.state == ("before", "before"):
            return "game"
        elif "after" in self.state:
            return "free"
        else:
            return "caution"

    def compute_pet(self):
        """Calculate PET considering physical sizes."""
        conflict_x = self.p_x
        conflict_y = self.v_y

        d_vehicle = conflict_x - self.v_x - self.ped_radius - self.vehicle_length / 2 - 0.2
        d_pedestrian = conflict_y - self.p_y - self.ped_radius - self.vehicle_width / 2 - 0.2

        if d_vehicle < -2 or d_pedestrian < -2:
            return np.inf

        t_vehicle = d_vehicle / self.vv_x if self.vv_x > 0 else np.inf
        t_pedestrian = d_pedestrian / self.vp_y if self.vp_y > 0 else np.inf
        return abs(t_pedestrian - t_vehicle)

    def compute_gamma(self, pet):
        if pet <= 1.770:
            return 3
        elif pet <= 4.962:
            return 2
        elif pet == np.inf:
            return 0
        else:
            return 1

    def generate_payoffs(self):
        """Compute payoff matrices for game."""
        pet = self.compute_pet()
        gamma_p = self.compute_gamma(pet)
        gamma_v = self.compute_gamma(pet)

        d_vehicle = self.p_x - self.v_x
        d_pedestrian = self.v_y - self.p_y
        d_offset_p, d_offset_v = 2, 2

        # Strategy payoffs
        p_11 = - gamma_p * self.theta_1 * np.exp(self.vp_y) + self.theta_3 * np.exp(1 / d_pedestrian + d_offset_p)
        v_11 = - gamma_v * self.theta_2 * np.exp(self.vv_x) + self.theta_3 * np.exp(1 / d_vehicle + d_offset_v)

        p_12 = gamma_p * self.theta_1 * np.exp(self.vp_y) - self.theta_4 * np.exp(self.tp)
        v_12 = gamma_v * self.theta_2 * np.exp(self.vv_x) + self.theta_3 * np.exp(1 / d_vehicle + d_offset_v)

        p_21 = gamma_p * self.theta_1 * np.exp(self.vp_y) + self.theta_3 * np.exp(1 / d_pedestrian + d_offset_p)
        v_21 = gamma_v * self.theta_2 * np.exp(self.vv_x) - self.theta_4 * np.exp(self.tv)

        p_22 = gamma_p * self.theta_1 * np.exp(self.vp_y) - self.theta_4 * np.exp(self.tp)
        v_22 = gamma_v * self.theta_2 * np.exp(self.vv_x) - self.theta_4 * np.exp(self.tv)

        vehicle_payoff = np.array([[v_11, v_12], [v_21, v_22]])
        pedestrian_payoff = np.array([[p_11, p_12], [p_21, p_22]])
        return vehicle_payoff, pedestrian_payoff

    def compute_mixed_strategy_equilibrium(self, vehicle_payoff, pedestrian_payoff):
        game = nash.Game(vehicle_payoff, pedestrian_payoff)
        equilibria = list(game.support_enumeration())
        for v_strat, p_strat in equilibria:
            if not np.allclose(v_strat, [0, 1]) and not np.allclose(v_strat, [1, 0]):
                return v_strat, p_strat
        return equilibria[0] if equilibria else (None, None)

    def update(self):
        """Top-level update called every frame."""
        mode = self.determine_state()

        if mode == "game":
            self.update_game()
        elif mode == "caution":
            self.update_caution()
        elif mode == "free":
            self.update_free()

    def update_game(self):
        """Use Nash equilibrium to update decisions."""
        vehicle_payoff, pedestrian_payoff = self.generate_payoffs()
        v_strat, p_strat = self.compute_mixed_strategy_equilibrium(vehicle_payoff, pedestrian_payoff)

        v_pass_prob = v_strat[0]
        self.v_pass_prob_history.append(v_pass_prob)
        p_pass_prob = p_strat[0]

        self.vv_x = v_pass_prob * self.vv_max if v_pass_prob >= 0.10 else 0
        self.vp_y = p_pass_prob * self.vp_max if p_pass_prob >= 0.15 else 0

        self.v_x += self.vv_x * self.dt
        self.p_y += self.vp_y * self.dt

        self.tv = 0.0 if self.vv_x >= 0.1 else self.tv + self.dt
        self.tp = 0.0 if self.vp_y >= 0.1 else self.tp + self.dt

    def update_caution(self):
        # Caution mode if one side of the game is in the conflict zone
        if self.state == ("before","in"):
            # Pedestrian is in the conflict zone
            if self.conflict_x - self.v_x < self.tresh_v + 0.1:
                self.vv_x = 0
        else:
            if self.conflict_y - self.p_y < self.tresh_p + 0.2:
                self.vp_y = 0

        self.v_x += self.vv_x * self.dt
        self.p_y += self.vp_y * self.dt

    def update_free(self):
        """Proceed with desired speed."""
        self.vv_x = self.vv_max
        self.vp_y = self.vp_max
        self.v_x += self.vv_x * self.dt
        self.p_y += self.vp_y * self.dt
