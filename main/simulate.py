import numpy as np
from amuse.lab import Particles, units
from amuse.units import nbody_system
from amuse.community.ph4.interface import ph4
from amuse.community.bhtree.interface import BHTree
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from make_ics import InitialConditions

def setup_particles_from_image(image_path, N, mass=1):
    ic = InitialConditions(image_path, N, MASS=mass)
    particles = Particles(N)
    particles.mass = ic.M | units.MSun
    pos_with_z = np.hstack((ic.POS, np.zeros((ic.N, 1))))  # Append zero z-coordinates
    particles.position = units.AU(pos_with_z)
    #particles.position = units.AU(ic.POS)
    vel_with_z = np.hstack((ic.VEL, np.zeros((ic.N, 1))))  # Add zero z-velocities, later will add depth perception for z
    particles.velocity = units.kms(vel_with_z)
    #particles.velocity = units.kms(ic.VEL)
    return particles, ic.colors

def evolve_system(particles, end_time, gravity_solver=BHTree):
    converter = nbody_system.nbody_to_si(particles.total_mass(), 1 | units.AU)  # Scale length 1 AU
    gravity = gravity_solver(converter)
    gravity.particles.add_particles(particles)

    positions = []
    times = np.arange(0, end_time.value_in(units.yr), 0.0028) | units.yr # Timestep of one day
    for time in times:
        gravity.evolve_model(time)
        positions.append(gravity.particles.copy())

    gravity.stop()
    return times, positions

def animate_system(positions, colors, filename='newton.mp4'):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('black')

    def update(frame):
        ax.clear()
        ax.set_facecolor('black')
        ax.scatter(positions[frame].x.value_in(units.AU),
                   positions[frame].y.value_in(units.AU),
                   color=colors,  # Use the passed colors
                   s=2)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f"Time: {frame * 0.0028:.1f} year")

    anim = FuncAnimation(fig, update, frames=len(positions), repeat=False)
    anim.save(filename, writer='ffmpeg', fps=5)
    plt.show()

if __name__ == "__main__":
    image_path = 'images/einstein.jpg'  
    N = 100000  #Number of particles
    mass = 1  # Total mass of each particles
    particles, colors = setup_particles_from_image(image_path, N, mass)
    end_time = 1 | units.yr  #Simulation end time
    times, positions = evolve_system(particles, end_time)
    animate_system(positions, colors)
