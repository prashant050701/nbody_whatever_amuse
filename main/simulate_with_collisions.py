import numpy as np
from amuse.lab import Particles, units
from amuse.units import nbody_system
from amuse.community.ph4.interface import ph4
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from make_ics import InitialConditions  # Make sure this import works as expected

def setup_particles_from_image(image_path, N, mass=1):
    ic = InitialConditions(image_path, N, MASS=mass)
    particles = Particles(N)
    particles.mass = ic.M | units.MSun
    pos_with_z = np.hstack((ic.POS, np.zeros((ic.N, 1))))  
    particles.position = units.AU(pos_with_z)
    vel_with_z = np.hstack((ic.VEL, np.zeros((ic.N, 1)))) 
    particles.velocity = units.kms(vel_with_z)
    particles.radius = 0.005 | units.AU  
    return particles, ic.colors

def merge_two_stars(particles, particles_in_encounter, tcoll):
    if len(particles_in_encounter) < 2:
        print("Not enough particles to merge.")
        return
    com_pos = particles_in_encounter.center_of_mass()
    com_vel = particles_in_encounter.center_of_mass_velocity()
    new_particle=Particles(1)
    new_particle.mass = particles_in_encounter.total_mass()
    new_particle.collision_time = tcoll
    new_particle.position = com_pos
    new_particle.velocity = com_vel
    new_particle.radius = particles_in_encounter.radius.sum()/len(particles_in_encounter.radius)
    particles.add_particles(new_particle)
    particles.remove_particles(particles_in_encounter)
    return new_particle


def evolve_system(particles, end_time, gravity_solver=ph4):
    converter = nbody_system.nbody_to_si(particles.total_mass(), 1 | units.AU)
    gravity = gravity_solver(converter, number_of_workers=1)
    gravity.particles.add_particles(particles)

    #channel_from_gd_to_framework = gravity.particles.new_channel_to(particles)
    stopping_condition = gravity.stopping_conditions.collision_detection
    stopping_condition.enable()
    t = []
    positions = []
    tcoll = []
    Nenc = 0
    times = np.arange(0, end_time.value_in(units.yr), 0.28) | units.yr
    for time in times:
        t.append(time.number)

        gravity.evolve_model(time)
        print(f"Evolving, current time: {time} yr")

        if stopping_condition.is_set():
            for ci in range(len(stopping_condition.particles(0))): 
                particles_in_encounter = Particles(
                    particles=[stopping_condition.particles(0)[ci],
                               stopping_condition.particles(1)[ci]])
                particles_in_encounter = particles_in_encounter.get_intersecting_subset_in(particles)

                new_particle = merge_two_stars(particles,
                                               particles_in_encounter,
                                               gravity.model_time)
                particles.synchronize_to(gravity.particles)
                Nenc+=1
                print("Resolve encounter number", Nenc)
                print("Collision at time=", time, new_particle.mass.sum(), \
                    new_particle.position.length(), "Nstars= ", len(particles), \
                    "Ncoll=", Nenc)

                pos = new_particle[0].position.number
                tcoll.append(time.number)
        positions.append(gravity.particles.copy())

        particles.move_to_center()
        #channel_from_gd_to_framework.copy()

    gravity.stop()
    return times, positions

def animate_system(positions, colors, filename='simulation.mp4'):
    fig, ax = plt.subplots(figsize=(8, 8))
    def update(frame):
        ax.clear()
        ax.scatter(positions[frame].x.value_in(units.AU),
                   positions[frame].y.value_in(units.AU),
                   color=colors[:len(positions[frame])], s=1)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f"Time: {frame * 0.28:.1f} years")
    anim = FuncAnimation(fig, update, frames=len(positions), repeat=False)
    anim.save(filename, writer='ffmpeg', fps=20)
    plt.show()

if __name__ == "__main__":
    image_path = 'images/einstein.jpg' 
    N = 5000  # Number of particles
    mass = 1  # Total mass of all particles
    particles, colors = setup_particles_from_image(image_path, N, mass)
    end_time = 5 | units.yr  # Simulation end time
    times, positions = evolve_system(particles, end_time)
    animate_system(positions, colors)
