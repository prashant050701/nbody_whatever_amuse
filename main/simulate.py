import numpy as np
from amuse.lab import Particles, units
from amuse.units import nbody_system
from amuse.community.ph4.interface import ph4
from amuse.community.bhtree.interface import BHTree
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
#from make_ics import InitialConditions
from make_ics_3d import InitialConditions #use this for depth perception

def setup_particles_from_image(image_path, N, mass=1):
    ic = InitialConditions(image_path, N, MASS=mass)
    particles = Particles(N)
    particles.mass = ic.M | units.MSun
    #pos_with_z = np.hstack((ic.POS, np.zeros((ic.N, 1)))) 
    #particles.position = units.AU(pos_with_z)
    particles.position = units.AU(ic.POS)
    #vel_with_z = np.hstack((ic.VEL, np.zeros((ic.N, 1))))  #gave zero z-velocities, later will add depth perception for z
    #particles.velocity = units.kms(vel_with_z)
    particles.velocity = units.kms(ic.VEL)
    return particles, ic.colors

def evolve_system(particles, end_time, gravity_solver=BHTree):
    converter = nbody_system.nbody_to_si(particles.total_mass(), 1 | units.AU)
    gravity = gravity_solver(converter)
    gravity.particles.add_particles(particles)

    positions = []
    times = np.arange(0, end_time.value_in(units.yr), 0.0028) | units.yr #approx one day timestep
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
                   color=colors,  #using the passed colors
                   s=2)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f"Time: {frame * 0.0028:.1f} year")

    anim = FuncAnimation(fig, update, frames=len(positions), repeat=False)
    anim.save(filename, writer='ffmpeg', fps=5)
    plt.show()

def animate_system_3d(positions, colors, filename='einstein_new_3d.mp4'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        x = positions[frame].x.value_in(units.AU)
        y = positions[frame].y.value_in(units.AU)
        z = positions[frame].z.value_in(units.AU)

        ax.scatter(x, y, z, color=colors, s=1)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)

        # Set labels and title
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        ax.set_title(f"Time: {frame * 0.028:.1f} year")

    anim = FuncAnimation(fig, update, frames=len(positions), repeat=False)
    anim.save(filename, writer='ffmpeg', fps=20)
    plt.show()

if __name__ == "__main__":
    image_path = 'sample_images/einstein.jpg'  
    N = 100000  #Number of particles
    mass = 1  # Total mass of each particles
    particles, colors = setup_particles_from_image(image_path, N, mass)
    end_time = 1 | units.yr  #Simulation end time
    times, positions = evolve_system(particles, end_time)
    animate_system(positions, colors)
    #animate_system_3d(positions, colors)
