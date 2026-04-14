use cgmath::Vector2;
use rand::Rng;

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct Particle {
    pub position: Vector2<f32>,
    pub velocity: Vector2<f32>,
    pub mass: f32,
    pub density: f32,
    pub pressure: f32,
}

#[allow(unused)]
pub struct SmoothedParticleHydrodynamicsSim {
    pub particles: Vec<Particle>,
    bounds: [f32; 2],
    smooth_radius: f32,
}

#[allow(unused)]
impl SmoothedParticleHydrodynamicsSim {
    pub fn new(
        nparticles: usize,
        bounds: [f32; 2],
        smooth_radius: f32,
    ) -> SmoothedParticleHydrodynamicsSim {
        let mut particles: Vec<Particle> = Vec::with_capacity(nparticles);
        let mut rng = rand::rng();

        for _ in 0..nparticles {
            let position = Vector2 {
                x: rng.random_range(0.0..bounds[0]),
                y: rng.random_range(0.0..bounds[1]),
            };
            let velocity = Vector2 { x: 0f32, y: 0f32 };
            let mass = 1f32;

            particles.push(Particle {
                position,
                velocity,
                mass,
                density: 0.0,
                pressure: 0.0,
            });
        }

        SmoothedParticleHydrodynamicsSim {
            particles,
            bounds,
            smooth_radius,
        }
    }

    pub fn update(&mut self, dt: f32) {
        for particle in &mut self.particles {
            particle.position += particle.velocity * dt;
        }

        for p_j in self.particles.clone() {
            for p_i in &mut self.particles {
                p_i.density +=
                    p_j.mass * kernel::poly6(p_i.position - p_j.position, self.smooth_radius);
            }
        }
    }
}

#[allow(unused)]
mod kernel {
    use cgmath::{InnerSpace, Vector2};
    use std::f32::consts::PI;

    pub fn poly6(r: Vector2<f32>, h: f32) -> f32 {
        (4.0 / (PI * h.powi(8))) * (h * h - r.magnitude2()).powi(3)
    }

    pub fn poly6_grad(r: Vector2<f32>, h: f32) -> Vector2<f32> {
        (-24.0 / (PI * h.powi(8))) * (h * h - r.magnitude2()).powi(2) * r
    }

    pub fn poly6_lapl(r: Vector2<f32>, h: f32) -> f32 {
        (-48.0 / (PI * h.powi(8))) * (h * h - r.magnitude2()) * (h * h - 3.0 * r.magnitude2())
    }

    pub fn spiky(r: Vector2<f32>, h: f32) -> f32 {
        (10.0 / (PI * h.powi(5))) * (h - r.magnitude()).powi(3)
    }

    pub fn spiky_grad(r: Vector2<f32>, h: f32) -> Vector2<f32> {
        (-30.0 / (PI * h.powi(5))) * (h - r.magnitude()).powi(2) * r.normalize()
    }

    pub fn spiky_lapl(r: Vector2<f32>, h: f32) -> f32 {
        (-30.0 / (PI * h.powi(5))) * (h - r.magnitude()) * (h - 3.0 * r.magnitude()) / r.magnitude()
    }
}
