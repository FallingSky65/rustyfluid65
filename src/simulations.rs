pub mod rigidbody;
pub mod rotate;
pub mod sph;

#[allow(dead_code)]
pub trait Simulation {
    fn sim_update(&mut self, dt: f32);
    fn get_instances(&self) -> Vec<crate::Instance>;
}

impl Simulation for rotate::RotateSim {
    fn sim_update(&mut self, dt: f32) {
        self.update(dt);
    }

    fn get_instances(&self) -> Vec<crate::Instance> {
        let instances: Vec<crate::Instance> = self
            .particles
            .iter()
            .map(|particle| crate::Instance {
                position: [
                    (particle.position[0] * 0.2 + 0.5) * crate::CANVAS_SIZE[0] as f32,
                    (particle.position[1] * 0.2 + 0.5) * crate::CANVAS_SIZE[1] as f32,
                    0.0,
                    0.0,
                ],
                color: [1.0, 1.0, 1.0, 1.0],
            })
            .collect();

        instances
    }
}

impl Simulation for rigidbody::RigidBodySim {
    fn sim_update(&mut self, dt: f32) {
        self.update(dt);
    }

    fn get_instances(&self) -> Vec<crate::Instance> {
        let instances: Vec<crate::Instance> = self
            .particles
            .iter()
            .map(|particle| crate::Instance {
                position: [particle.position.x, particle.position.y, 0.0, 0.0],
                color: [particle.color[0], particle.color[1], particle.color[2], 1.0],
            })
            .collect();

        instances
    }
}
