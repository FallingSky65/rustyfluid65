use cgmath::{InnerSpace, Vector2};
use rand::Rng;

#[allow(dead_code)]
pub struct Particle {
    pub position: Vector2<f32>,
    velocity: Vector2<f32>,
    radius: f32,
    pub color: [f32; 3],
}

pub struct RigidBodySim {
    pub particles: Vec<Particle>,
    bounds: [f32; 2],
}

fn hsv2rgb(hsv: [f32; 3]) -> [f32; 3] {
    let [h, s, v] = hsv;
    let i = (h.clamp(0.0, 1.0) * 6.0).floor() as i32;
    let f = (h.clamp(0.0, 1.0) * 6.0).fract();

    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        5 => [v, p, q],
        _ => [0.0, 0.0, 0.0],
    }
}

#[allow(unused)]
impl RigidBodySim {
    pub fn new(nparticles: usize, bounds: [f32; 2], radius: f32) -> RigidBodySim {
        let mut particles: Vec<Particle> = Vec::with_capacity(nparticles);
        let mut rng = rand::rng();

        for _ in 0..nparticles {
            let position = Vector2 {
                x: rng.random_range(radius..bounds[0] - radius),
                y: rng.random_range(radius..bounds[1] - radius),
            };
            let velocity = Vector2 {
                x: rng.sample(rand_distr::StandardNormal),
                y: rng.sample(rand_distr::StandardNormal),
            };
            particles.push(Particle {
                position,
                velocity,
                radius,
                color: hsv2rgb([rng.sample(rand_distr::StandardUniform), 1.0, 1.0]),
            });
        }

        for _ in 0..16 {
            for i in 1..particles.len() {
                for j in 0..i {
                    let (s1, s2) = particles.split_at_mut(i);
                    let p1 = &mut s1[j];
                    let p2 = &mut s2[0];

                    if (p1.position - p2.position).magnitude() > (p1.radius + p2.radius) {
                        continue;
                    }

                    let p_update = (p1.position - p2.position).normalize()
                        * (p1.radius + p2.radius - (p1.position - p2.position).magnitude());

                    p1.position += p_update;
                    p2.position -= p_update;
                }
            }
        }

        RigidBodySim { particles, bounds }
    }

    pub fn update(&mut self, dt: f32) {
        // https://www.cs.ubc.ca/~rhodin/2020_2021_CPSC_427/lectures/D_CollisionTutorial.pdf

        for particle in &mut self.particles {
            particle.position += particle.velocity * 50.0 * dt;
            particle.velocity.y -= 20.0 * dt;
        }

        // accumulated impulse
        //let mut accJ: HashMap<(usize, usize), >

        for i in 1..self.particles.len() {
            for j in 0..i {
                let (s1, s2) = self.particles.split_at_mut(i);
                let p1 = &mut s1[j];
                let p2 = &mut s2[0];

                if (p1.position - p2.position).magnitude() > (p1.radius + p2.radius) {
                    continue;
                }

                // update based on elastic collision, conservation of momentum
                let v_update = (p1.velocity - p2.velocity).dot(p1.position - p2.position)
                    / (p1.position - p2.position).magnitude2()
                    * (p1.position - p2.position);

                let _v_update2 =
                    (p1.velocity - p2.velocity).dot((p1.position - p2.position).normalize());

                p1.velocity -= v_update;
                p2.velocity += v_update;

                let p_update = (p1.position - p2.position).normalize()
                    * (p1.radius + p2.radius - (p1.position - p2.position).magnitude() + 0.01);

                p1.position += p_update;
                p2.position -= p_update;
            }
        }

        for particle in &mut self.particles {
            if particle.position.x < particle.radius
                || particle.position.x > self.bounds[0] - particle.radius
            {
                particle.position.x -= particle.velocity.x * 50.0 * dt;
                particle.velocity.x *= -0.9;
            }

            if particle.position.y < particle.radius
                || particle.position.y > self.bounds[1] - particle.radius
            {
                particle.position.y -= particle.velocity.y * 50.0 * dt;
                particle.velocity.y *= -0.9;
            }
        }
    }
}
