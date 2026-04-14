use rand::Rng;

pub struct Particle {
    pub position: [f32; 2],
}

pub struct RotateSim {
    pub particles: Vec<Particle>,
    pub speed: f32,
}

#[allow(unused)]
impl RotateSim {
    pub fn new(nparticles: usize, speed: f32) -> RotateSim {
        let mut particles: Vec<Particle> = Vec::with_capacity(nparticles);
        let mut rng = rand::rng();

        for _ in 0..nparticles {
            particles.push(Particle {
                position: [
                    rng.sample(rand_distr::StandardNormal),
                    rng.sample(rand_distr::StandardNormal),
                ],
            });
        }

        RotateSim { particles, speed }
    }

    pub fn update(&mut self, dt: f32) {
        let dtheta = self.speed * dt;
        let (s, c) = dtheta.sin_cos();

        for particle in &mut self.particles {
            particle.position = [
                particle.position[0] * c - particle.position[1] * s,
                particle.position[0] * s + particle.position[1] * c,
            ]
        }
    }
}
