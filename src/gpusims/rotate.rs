use rand::Rng;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 2],
}

#[allow(unused)]
pub struct GPURotateSim {
    pipeline: wgpu::ComputePipeline,
    particle_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub n: usize,
}

#[allow(unused)]
impl GPURotateSim {
    pub fn new(device: &wgpu::Device, n: usize) -> GPURotateSim {
        let shader = device.create_shader_module(wgpu::include_wgsl!("rotate.wgsl"));

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let mut particles: Vec<Particle> = Vec::with_capacity(n);
        let mut rng = rand::rng();

        for _ in 0..n {
            particles.push(Particle {
                position: [
                    rng.sample(rand_distr::StandardNormal),
                    rng.sample(rand_distr::StandardNormal),
                ],
            });
        }

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: (std::mem::size_of::<crate::Instance>() * n) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        });

        return GPURotateSim {
            pipeline,
            particle_buffer,
            instance_buffer,
            bind_group,
            n,
        };
    }

    pub fn update(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&Default::default());

        {
            let num_dispatches = self.n.div_ceil(64) as u32;

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_dispatches, 1, 1);
        }

        queue.submit([encoder.finish()]);
    }
}
