import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.particle as pu
from genesis.repr_base import RBC


class Emitter(RBC):
    """
    A particle emitter for fluid or material simulation.

    The Emitter manages the generation of particles into the simulation domain, allowing directional or omnidirectional
    emissions with various droplet shapes. It supports resetting, shape-based emission, and spherical omni-emission.

    Parameters
    ----------
    max_particles : int
        The maximum number of particles that this emitter can handle.
    """

    def __init__(self, max_particles):
        self._uid = gs.UID()
        self._entity = None

        self._max_particles = max_particles

        self._acc_droplet_len = 0.0  # accumulated droplet length to be emitted

        gs.logger.info(
            f"Creating ~<{self.__repr_name__()}>~. id: ~~~<{self._uid}>~~~, max_particles: ~<{max_particles}>~."
        )

    def set_entity(self, entity):
        """
        Assign an entity to the emitter and initialize relevant simulation and solver references.

        Parameters
        ----------
        entity : Entity
            The entity to associate with the emitter. This entity should contain the solver, simulation context, and particle sampler.
        """
        pass

    def reset(self):
        """
        Reset the emitter's internal particle index to start emitting from the beginning.
        """
        self._next_particle = 0

    def emit(
        self,
        droplet_shape,
        droplet_size,
        droplet_length=None,
        pos=(0.5, 0.5, 1.0),
        direction=(0, 0, -1),
        theta=0.0,
        speed=1.0,
        p_size=None,
    ):
        """
        Emit particles in a specified shape and direction from a nozzle.

        Parameters
        ----------
        droplet_shape : str
            The shape of the emitted droplet. Options: "circle", "sphere", "square", "rectangle".
        droplet_size : float or tuple
            Size of the droplet. A single float for symmetric shapes, or a tuple of (width, height) for rectangles.
        droplet_length : float, optional
            Length of the droplet in the emitting direction. If None, calculated from speed and simulation timing.
        pos : tuple of float
            World position of the nozzle from which the droplet is emitted.
        direction : tuple of float
            Direction vector of the emitted droplet.
        theta : float
            Rotation angle (in radians) around the droplet axis.
        speed : float
            Emission speed of the particles.
        p_size : float, optional
            Particle size used for filling the droplet. Defaults to the solver's particle size.

        Raises
        ------
        Exception
            If the shape is unsupported or the emission would place particles outside the simulation boundary.
        """
        pass

    def emit_omni(self, source_radius=0.1, pos=(0.5, 0.5, 1.0), speed=1.0, particle_size=None):
        """
        Use a sphere-shaped source to emit particles in all directions.

        Parameters:
        ----------
        source_radius: float, optional
            The radius of the sphere source. Particles will be emitted from a shell with inner radius using
            '0.8 * source_radius' and outer radius using source_radius.
        pos: array_like, shape=(3,)
            The center of the sphere source.
        speed: float
            The speed of the emitted particles.
        particle_size: float | None
            The size (diameter) of the emitted particles. The actual number of particles emitted is determined by the
            volume of the sphere source and the size of the particles. If None, the solver's particle size is used.
            Note that this particle size only affects computation for number of particles emitted, not the actual size
            of the particles in simulation and rendering.
        """
        pass

    @property
    def uid(self):
        """The unique identifier of the emitter."""
        pass

    @property
    def entity(self):
        """The entity associated with the emitter."""
        pass

    @property
    def max_particles(self):
        """The maximum number of particles this emitter can emit."""
        pass

    @property
    def solver(self):
        """The solver used by the emitter's associated entity."""
        pass

    @property
    def next_particle(self):
        """The index of the next particle to be emitted."""
        pass
