use rand::Rng;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UBO {
    pub nparticles: u32,
    pub smooth_radius: f32,
    pub block_size: u32,
    pub flip: u32,
    pub gas_constant: f32,
    pub target_density: f32,
    pub canvas_width: f32,
    pub canvas_height: f32,

    pub dt: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    acc: [f32; 2],
    normal: [f32; 2],

    mass: f32,
    density: f32,
    pressure: f32,
    _pad: f32,
}

#[allow(unused)]
pub struct GPUSmoothedParticleHydrodynamicsSim {
    // compute pipelines + bind groups
    calc_hash: wgpu::ComputePipeline,
    bitonic_sort: wgpu::ComputePipeline,
    find_range: wgpu::ComputePipeline,
    set_position: wgpu::ComputePipeline,
    set_density_pressure: wgpu::ComputePipeline,
    set_normal: wgpu::ComputePipeline,
    set_acceleration: wgpu::ComputePipeline,
    move_and_collide: wgpu::ComputePipeline,

    particle_buffer: wgpu::Buffer,
    particle_hash: wgpu::Buffer,
    cell_range: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    pub n: usize,
    ubo: UBO,

    bitonic4096: wgpu::ComputePipeline,
    hash_bind: wgpu::BindGroup,
}

#[allow(unused)]
impl GPUSmoothedParticleHydrodynamicsSim {
    pub fn new(device: &wgpu::Device, total_mass: f32, box_radius: f32, ubo: UBO) -> Self {
        let mut ubo = ubo;
        let n = ubo.nparticles as usize;

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::cast_slice(&[ubo]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let mut rng = rand::rng();
        let distr_x = rand_distr::Uniform::new(-1.0 * box_radius, 1.0 * box_radius).unwrap();
        let distr_y = rand_distr::Uniform::new(-box_radius, -0.4 * box_radius).unwrap();
        let particles: Vec<Particle> = (0..n)
            .map(|i| Particle {
                pos: [rng.sample(distr_x), rng.sample(distr_y)],
                vel: [0.0, 0.0],
                acc: [0.0, 0.0],
                normal: [0.0, 0.0],
                mass: total_mass / n as f32,
                density: Default::default(),
                pressure: Default::default(),
                _pad: Default::default(),
            })
            .collect();

        println!("n: {}", n);
        println!("number of particles: {}", particles.len());

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle buffer"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let particle_hash = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle hash"),
            size: (std::mem::size_of::<[u32; 2]>() * n) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let cell_range = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell range"),
            size: (std::mem::size_of::<[u32; 2]>() * n) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance buffer"),
            size: (std::mem::size_of::<crate::Instance>() * n) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_hash.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_range.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline layout"),
            bind_group_layouts: &[Some(&uniform_bind_group_layout), Some(&bind_group_layout)],
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("sph.wgsl"));

        let calc_hash = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("calc hash"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("calc_hash"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let bitonic_sort = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bitonic sort"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("bitonic_sort"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let find_range = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("find range"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("find_range"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let set_position = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("set position"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("set_position"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let set_density_pressure =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("set density pressure"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("set_density_pressure"),
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        let set_normal = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("set normal"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("set_normal"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let set_acceleration = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("set acceleration"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("set_acceleration"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let move_and_collide = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("move and collide"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("move_and_collide"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let bitonic4096shader =
            device.create_shader_module(wgpu::include_wgsl!("bitonic4096.wgsl"));
        let bitonic4096 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bitonic sort 4096"),
            layout: None,
            module: &bitonic4096shader,
            entry_point: Some("bitonic_sort4096"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });
        let hash_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_hash bind group"),
            layout: &bitonic4096.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_hash.as_entire_binding(),
            }],
        });

        Self {
            calc_hash,
            bitonic_sort,
            find_range,
            set_position,
            set_density_pressure,
            set_normal,
            set_acceleration,
            move_and_collide,
            particle_buffer,
            particle_hash,
            cell_range,
            instance_buffer,
            bind_group,
            uniform_buffer,
            uniform_bind_group,
            n,
            ubo,

            bitonic4096,
            hash_bind,
        }
    }

    pub fn update(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &wgpu_profiler::GpuProfiler,
    ) {
        let mut scope = profiler.scope("SPH Compute", encoder);
        //encoder.write_timestamp(query_set, 0);

        let num_dispatches = self.n.div_ceil(64) as u32;

        let mut ubo = self.ubo;
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[ubo]));
        {
            let mut pass = scope.scoped_compute_pass("calc_hash");
            pass.set_pipeline(&self.calc_hash);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_dispatches, 1, 1);
        }

        // {
        //     let mut pass = scope.scoped_compute_pass("bitonic4096");
        //     pass.set_pipeline(&self.calc_hash);
        //     pass.set_pipeline(&self.bitonic4096);
        //     pass.set_bind_group(0, &self.hash_bind, &[]);
        //     pass.dispatch_workgroups(1, 1, 1);
        // }

        // let mut bitonic_passes: u32 = 0;
        {
            let mut bitonic_scope = scope.scope("bitonic_sort");
            let numitems = self.n.next_power_of_two() as u32;
            let numworkers = numitems / 2;
            let mut i: u32 = 2;
            while i <= numitems {
                let mut i_scope = bitonic_scope.scope(format!("i={}", i));
                ubo.block_size = i;
                ubo.flip = 1;
                queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[ubo]));
                {
                    let mut pass = i_scope.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.bitonic_sort);
                    pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    pass.set_bind_group(1, &self.bind_group, &[]);
                    pass.dispatch_workgroups(numworkers.div_ceil(64), 1, 1);

                    // bitonic_passes += 1;
                }

                ubo.flip = 0;
                let mut j: u32 = i / 2;
                while j >= 2 {
                    ubo.block_size = j;
                    queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[ubo]));
                    {
                        let mut pass = i_scope.scoped_compute_pass(format!("j={}", j));
                        pass.set_pipeline(&self.bitonic_sort);
                        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                        pass.set_bind_group(1, &self.bind_group, &[]);
                        pass.dispatch_workgroups(numworkers.div_ceil(64), 1, 1);

                        // bitonic_passes += 1;
                    }

                    j /= 2;
                }

                i *= 2;
            }
        }

        // println!("{} bitonic passes", bitonic_passes);

        {
            let mut pass = scope.scoped_compute_pass("find_range");
            pass.set_pipeline(&self.find_range);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_dispatches, 1, 1);
        }

        {
            let mut pass = scope.scoped_compute_pass("set_position");
            pass.set_pipeline(&self.set_position);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_dispatches, 1, 1);
        }

        {
            let mut pass = scope.scoped_compute_pass("set_density_pressure");
            pass.set_pipeline(&self.set_density_pressure);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_dispatches, 1, 1);
        }

        {
            let mut pass = scope.scoped_compute_pass("set_normal");
            pass.set_pipeline(&self.set_normal);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_dispatches, 1, 1);
        }

        {
            let mut pass = scope.scoped_compute_pass("set_acceleration");
            pass.set_pipeline(&self.set_acceleration);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_dispatches, 1, 1);
        }

        {
            let mut pass = scope.scoped_compute_pass("move_and_collide");
            pass.set_pipeline(&self.move_and_collide);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_dispatches, 1, 1);
        }
    }
}
