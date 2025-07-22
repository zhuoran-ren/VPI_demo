import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from VPI import ChickenGame

def animate_simulation(game, interval=100, steps=100):
    """
    Visualize the Chicken Game simulation using matplotlib animation
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.set_title("Pedestrian-Vehicle Interaction Simulation")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")

    # Draw road and zebra crossing
    draw_crosswalk(ax)
    draw_road_background(ax)

    # Arrows for vehicle and pedestrian
    vehicle_arrow = patches.FancyArrowPatch(
        posA=(game.v_x, game.v_y),              
        posB=(game.v_x + 0.4, game.v_y),        
        arrowstyle='->',
        color='cyan',
        mutation_scale=10
    )
    pedestrian_arrow = patches.FancyArrowPatch(
        posA=(game.p_x, game.p_y),
        posB=(game.p_x, game.p_y + 0.4),
        arrowstyle='->',
        color='orange',
        mutation_aspect=10
    )

    vehicle_patch = patches.Rectangle((game.v_x - 0.4, game.v_y - 0.2), 0.8, 0.4, color='blue', label='Vehicle')
    pedestrian_patch = plt.Circle((game.p_x, game.p_y), 0.1, color='red', label='Pedestrian')

    ax.add_patch(vehicle_patch)
    ax.add_patch(pedestrian_patch)
    ax.add_patch(vehicle_arrow)
    ax.add_patch(pedestrian_arrow)

    def init():
        vehicle_patch.set_xy((game.v_x - 0.4, game.v_y - 0.2))
        pedestrian_patch.center = (game.p_x, game.p_y)
        
        start_vehicle = (game.v_x, game.v_y)
        end_vehicle = (game.v_x + 0.4, game.v_y)
        vehicle_arrow.set_positions(start_vehicle, end_vehicle)

        start_ped = (game.p_x, game.p_y)
        end_ped = (game.p_x, game.p_y + 0.4)
        pedestrian_arrow.set_positions(start_ped, end_ped)
        
        return vehicle_patch, pedestrian_patch, vehicle_arrow, pedestrian_arrow
    
    def update(frame):

        # Perform game update
        game.update()

        # Update vehicle rectangle position
        vehicle_patch.set_xy((game.v_x - 0.4, game.v_y - 0.2))
        # Update pedestrian circle position
        pedestrian_patch.center = (game.p_x, game.p_y)

        start_vehicle = (game.v_x, game.v_y)
        end_vehicle = (game.v_x + 0.4, game.v_y)
        vehicle_arrow.set_positions(start_vehicle, end_vehicle)

        start_ped = (game.p_x, game.p_y)
        end_ped = (game.p_x, game.p_y + 0.4)
        pedestrian_arrow.set_positions(start_ped, end_ped)

        return vehicle_patch, pedestrian_patch, vehicle_arrow, pedestrian_arrow

    ani = animation.FuncAnimation(
        fig, update, frames=steps,
        init_func=init, blit=True, interval=interval
    )

    plt.show()
    
# draw background helpers
def draw_road_background(ax):
    road = patches.Rectangle((0, 2), 10, 2, color='#A9A9A9', zorder=0)
    ax.add_patch(road)

def draw_crosswalk(ax, x_start=4.5, x_end=5.5, y_start=2, y_end=4, stripe_width=0.2, gap=0.2):
    y = y_start
    while y < y_end:
        stripe = patches.Rectangle((x_start, y), x_end - x_start, stripe_width, color='white', zorder=1)
        ax.add_patch(stripe)
        y += stripe_width + gap

if __name__ == "__main__":
    game = ChickenGame()
    animate_simulation(game)