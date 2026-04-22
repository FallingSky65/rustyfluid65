use rand::Rng;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UBO {
    pub nparticles: u32,
    pub smooth_radius: f32,
    pub gas_constant: f32,
    pub target_density: f32,
    pub canvas_width: f32,
    pub canvas_height: f32,

    pub dt: f32,

    pub mouse_x: f32,
    pub mouse_y: f32,
    pub mouse_state: u32,
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

const COMPILATION_OPTIONS: wgpu::PipelineCompilationOptions<'_> = wgpu::PipelineCompilationOptions {
    constants: &[
        ("WORKGROUP_SIZE", crate::WORKGROUP_SIZE as f64)
    ],
    zero_initialize_workgroup_memory: false,
};

struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    num_dispatches: u32,
    scope_name: Option<String>
}

impl Pipeline {
    fn new(
        device: &wgpu::Device,
        shader: wgpu::ShaderModuleDescriptor<'_>,
        bindings: Vec<wgpu::BindingResource<'_>>,
        num_dispatches: u32,
        entry_point: Option<&str>,
        compilation_options: Option<wgpu::PipelineCompilationOptions>,
        label: Option<&str>,
        scope_name: Option<String>,
    ) -> Self {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label,
            layout: None,
            module: &device.create_shader_module(shader),
            entry_point,
            compilation_options: if let Some(opts) = compilation_options {
                opts
            } else {
                Default::default()
            },
            cache: Default::default(),
        });
        let entries: Vec<wgpu::BindGroupEntry> = bindings
            .into_iter()
            .enumerate()
            .map(|(i, binding)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: binding
            })
            .collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: entries.as_slice(),
        });

        return Self { pipeline, bind_group, num_dispatches, scope_name }
    }

    fn new_with_shifting_ubo(
        device: &wgpu::Device,
        shader: wgpu::ShaderModuleDescriptor<'_>,
        layout: Option<&wgpu::PipelineLayout>,
        bindings: Vec<wgpu::BindingResource<'_>>,
        num_dispatches: u32,
        entry_point: Option<&str>,
        compilation_options: Option<wgpu::PipelineCompilationOptions>,
        label: Option<&str>,
        scope_name: Option<String>,
    ) -> Self {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label,
            layout,
            module: &device.create_shader_module(shader),
            entry_point,
            compilation_options: if let Some(opts) = compilation_options {
                opts
            } else {
                Default::default()
            },
            cache: Default::default(),
        });
        let entries: Vec<wgpu::BindGroupEntry> = bindings
            .into_iter()
            .enumerate()
            .map(|(i, binding)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: binding
            })
            .collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: entries.as_slice(),
        });

        return Self { pipeline, bind_group, num_dispatches, scope_name };
    }

    fn do_pass(&self, scope: &mut wgpu_profiler::Scope<wgpu::CommandEncoder>) {
        if let Some(scope_name) = &self.scope_name {
            let mut pass = scope.scoped_compute_pass(scope_name);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.num_dispatches, 1, 1);
        } else {
            let mut pass = scope.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.num_dispatches, 1, 1);
        }
    }
    
    fn do_pass_offset_ubo(&self, scope: &mut wgpu_profiler::Scope<wgpu::CommandEncoder>, ubo_bind: &wgpu::BindGroup, index: u32) {
        if let Some(scope_name) = &self.scope_name {
            let mut pass = scope.scoped_compute_pass(scope_name);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, ubo_bind, &[index * 256]);
            pass.dispatch_workgroups(self.num_dispatches, 1, 1);
        } else {
            let mut pass = scope.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, ubo_bind, &[index * 256]);
            pass.dispatch_workgroups(self.num_dispatches, 1, 1);
        }
    }
}

#[allow(unused)]
fn readback_buffer(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, label: &str) {
    let mut encoder = device.create_command_encoder(&Default::default());
    let size = src.size();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));
    
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None
    });
    rx.recv().unwrap().unwrap();
    
    let data = slice.get_mapped_range();
    let values: &[u32] = bytemuck::cast_slice(&data);
    println!("=== {} ===", label);
    for (i, v) in values.iter().enumerate() {
        print!("{:4}: {:8}  ", i, v);
        if (i + 1) % 2 == 0 { println!(); }
    }
    println!();
}

#[allow(unused)]
pub struct RadixSort {
    local_scan: Pipeline,
    reduce: Pipeline,
    downsweep: Pipeline,
    scatter: Pipeline,
    ubo: wgpu::Buffer,
    ubo_bind: wgpu::BindGroup,
    block_sums: wgpu::Buffer,
    local_prefix: wgpu::Buffer,
    global_prefix: wgpu::Buffer,
    aux: wgpu::Buffer,

    n: u32,
    num_blocks: u32,
}

impl RadixSort {
    fn new(device: &wgpu::Device, hashes1: &wgpu::Buffer, hashes2: &wgpu::Buffer, n: u32) -> Self {
        let num_blocks = n.div_ceil(256);

        let ubo_contents: [u32; 64 * 16] = std::array::from_fn(|i| match i % 64 {
            0 => n as u32,
            1 => (i as u32 / 64) * 2, // shift
            2 => num_blocks as u32,
            3 => (i as u32 / 64) % 2, // pingpong
            _ => 0
        });

        let ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("radix ubo"),
            contents: bytemuck::cast_slice(&[ubo_contents]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let ubo_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix ubo layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(256),
                },
                visibility: wgpu::ShaderStages::COMPUTE,
            }]
        });
        let ubo_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix ubo bind"),
            layout: &ubo_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &ubo,
                    offset: 0,
                    size: wgpu::BufferSize::new(256),
                })
            }]
        });

        let block_sums = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block sums"),
            size: 16 * num_blocks as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let local_prefix = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("local prefix"),
            size: 4 * n as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let global_prefix = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global prefix"),
            size: 16 * num_blocks as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let aux = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aux"),
            size: 4 * 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let local_scan = Pipeline::new_with_shifting_ubo(
            device,
            wgpu::include_wgsl!("radix_sort/local_scan.wgsl"),
            Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("local scan layout"),
                bind_group_layouts: &[
                    Some(&Self::get_bind_layout(device, vec![false, false, false, false])),
                    Some(&ubo_layout),
                ],
                immediate_size: 0
            })),
            vec![
                hashes1.as_entire_binding(),
                hashes2.as_entire_binding(),
                block_sums.as_entire_binding(),
                local_prefix.as_entire_binding(),
            ],
            num_blocks,
            Some("local_scan"),
            None,
            Some("local scan"),
            None
        );
        
        let global_scan_shader = wgpu::include_wgsl!("radix_sort/global_scan.wgsl");
        let reduce = Pipeline::new_with_shifting_ubo(
            device,
            global_scan_shader.clone(),
            Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("local scan layout"),
                bind_group_layouts: &[
                    Some(&Self::get_bind_layout(device, vec![false, false, false])),
                    Some(&ubo_layout),
                ],
                immediate_size: 0
            })),
            vec![
                global_prefix.as_entire_binding(),
                aux.as_entire_binding(),
                block_sums.as_entire_binding(),
            ],
            (4 * num_blocks).div_ceil(256),
            Some("reduce"),
            None,
            Some("reduce"),
            None
        );
        let downsweep = Pipeline::new_with_shifting_ubo(
            device,
            global_scan_shader,
            Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("local scan layout"),
                bind_group_layouts: &[
                    Some(&Self::get_bind_layout(device, vec![false, false])),
                    Some(&ubo_layout),
                ],
                immediate_size: 0
            })),
            vec![
                global_prefix.as_entire_binding(),
                aux.as_entire_binding(),
            ],
            1,
            Some("downsweep"),
            None,
            Some("downsweep"),
            None
        );

        let scatter = Pipeline::new_with_shifting_ubo(
            device,
            wgpu::include_wgsl!("radix_sort/scatter.wgsl"),
            Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("local scan layout"),
                bind_group_layouts: &[
                    Some(&Self::get_bind_layout(device, vec![false, false, false, false])),
                    Some(&ubo_layout),
                ],
                immediate_size: 0
            })),
            vec![
                hashes1.as_entire_binding(),
                hashes2.as_entire_binding(),
                local_prefix.as_entire_binding(),
                global_prefix.as_entire_binding(),
            ],
            num_blocks,
            Some("scatter"),
            None,
            Some("scatter"),
            None
        );
    
        return Self { local_scan, reduce, downsweep, scatter, ubo, ubo_bind, block_sums, local_prefix, global_prefix, aux, n, num_blocks };
    }

    fn get_bind_layout(device: &wgpu::Device, read_only: Vec<bool>) -> wgpu::BindGroupLayout {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: *r },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                visibility: wgpu::ShaderStages::COMPUTE,
            }).collect();

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("local scan bind layout"),
            entries: entries.as_slice()
        })
    }
    
    fn do_sort(&self, scope: &mut wgpu_profiler::Scope<wgpu::CommandEncoder>) {
        let mut radix_scope = scope.scope("radix_sort");
        for i in 0..8u32 {
            self.local_scan.do_pass_offset_ubo(&mut radix_scope, &self.ubo_bind, i);
            self.reduce.do_pass_offset_ubo(&mut radix_scope, &self.ubo_bind, i);
            self.downsweep.do_pass_offset_ubo(&mut radix_scope, &self.ubo_bind, i);
            self.scatter.do_pass_offset_ubo(&mut radix_scope, &self.ubo_bind, i);
        }
    }

    #[allow(unused)]
    pub fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue, hashes1: &wgpu::Buffer, hashes2: &wgpu::Buffer) {
        readback_buffer(device, queue, &self.block_sums, "block_sums");
        readback_buffer(device, queue, &self.local_prefix, "local_prefix");
        readback_buffer(device, queue, &self.global_prefix, "global_prefix");
        readback_buffer(device, queue, &self.aux, "aux");
        readback_buffer(device, queue, hashes1, "hashes1");
        readback_buffer(device, queue, hashes2, "hashes2");
    }
}

struct ParticleBuffers {
    pos: wgpu::Buffer,
    vel: wgpu::Buffer,
    acc: wgpu::Buffer,
    mass: wgpu::Buffer,
    normal: wgpu::Buffer,
    density: wgpu::Buffer,
}

impl ParticleBuffers {
    fn new(device: &wgpu::Device, particles: Vec<Particle>) -> Self {
        let positions: Vec<[f32; 2]> = particles.iter().map(|p| p.pos).collect();
        let pos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pos"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });
        
        let velocities: Vec<[f32; 2]> = particles.iter().map(|p| p.vel).collect();
        let vel = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vel"),
            contents: bytemuck::cast_slice(&velocities),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });
        
        let accelerations: Vec<[f32; 2]> = particles.iter().map(|p| p.acc).collect();
        let acc = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("acc"),
            contents: bytemuck::cast_slice(&accelerations),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });
        
        let masses: Vec<f32> = particles.iter().map(|p| p.mass).collect();
        let mass = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mass"),
            contents: bytemuck::cast_slice(&masses),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });
        
        let normal = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("normal"),
            size: (std::mem::size_of::<[f32; 2]>() * particles.len()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let density = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("density"),
            size: (std::mem::size_of::<f32>() * particles.len()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        return Self { pos, vel, acc, mass, normal, density }
    }
}

#[allow(unused)]
pub struct GPUSmoothedParticleHydrodynamicsSim {
    pbuffers: ParticleBuffers,
    hashes1: wgpu::Buffer,
    hashes2: wgpu::Buffer,
    cell_range: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,

    calc_hash: Pipeline,
    find_range: Pipeline,
    set_density: Pipeline,
    set_normal: Pipeline,
    set_acceleration: Pipeline,
    move_and_collide: Pipeline,

    radix_sort: RadixSort,

    pub n: usize,
    ubo: UBO,

    control_info: crate::ControlInfo,
}

#[allow(unused)]
impl GPUSmoothedParticleHydrodynamicsSim {
    pub fn new(device: &wgpu::Device, total_mass: f32, box_radius: f32, ubo: UBO) -> Self {
        let mut ubo = ubo;
        let n = ubo.nparticles as usize;
        let num_dispatches = n.div_ceil(256) as u32;
        
        let mut rng = rand::rng();
        let distr_x = rand_distr::Uniform::new(-1.0 * box_radius, 0.0 * box_radius).unwrap();
        let distr_y = rand_distr::Uniform::new(-box_radius, box_radius).unwrap();
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

        println!("number of particles: {}", particles.len());

        let pbuffers = ParticleBuffers::new(device, particles);
        
        let hashes1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hashes1"),
            size: (std::mem::size_of::<[u32; 2]>() * n) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let hashes2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hashes2"),
            size: (std::mem::size_of::<[u32; 2]>() * n) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let cell_range = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell range"),
            size: (std::mem::size_of::<[u32; 2]>() * n) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance buffer"),
            size: (std::mem::size_of::<crate::Instance>() * n) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::cast_slice(&[ubo]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });


        let calc_hash = Pipeline::new(
            device, 
            wgpu::include_wgsl!("sph/calc_hash.wgsl"), 
            vec![
                uniform_buffer.as_entire_binding(),
                pbuffers.pos.as_entire_binding(),
                hashes1.as_entire_binding(),
            ], 
            num_dispatches, 
            Some("calc_hash"), 
            Some(COMPILATION_OPTIONS),
            Some("calc hash"), 
            Some("calc_hash".to_string())
        );

        let find_range = Pipeline::new(
            device, 
            wgpu::include_wgsl!("sph/find_range.wgsl"), 
            vec![
                uniform_buffer.as_entire_binding(),
                hashes1.as_entire_binding(),
                cell_range.as_entire_binding(),
            ], 
            num_dispatches, 
            Some("find_range"), 
            Some(COMPILATION_OPTIONS),
            Some("find range"), 
            Some("find_range".to_string())
        );

        let set_density = Pipeline::new(
            device, 
            wgpu::include_wgsl!("sph/set_density.wgsl"), 
            vec![
                uniform_buffer.as_entire_binding(),
                hashes1.as_entire_binding(),
                cell_range.as_entire_binding(),
                pbuffers.pos.as_entire_binding(),
                pbuffers.mass.as_entire_binding(),
                pbuffers.density.as_entire_binding(),
            ], 
            num_dispatches, 
            Some("set_density"), 
            Some(COMPILATION_OPTIONS),
            Some("set density"), 
            Some("set_density".to_string())
        );
        
        let set_normal = Pipeline::new(
            device, 
            wgpu::include_wgsl!("sph/set_normal.wgsl"), 
            vec![
                uniform_buffer.as_entire_binding(),
                hashes1.as_entire_binding(),
                cell_range.as_entire_binding(),
                pbuffers.pos.as_entire_binding(),
                pbuffers.mass.as_entire_binding(),
                pbuffers.density.as_entire_binding(),
                pbuffers.normal.as_entire_binding(),
            ], 
            num_dispatches, 
            Some("set_normal"), 
            Some(COMPILATION_OPTIONS),
            Some("set normal"), 
            Some("set_normal".to_string())
        );

        let set_acceleration = Pipeline::new(
            device, 
            wgpu::include_wgsl!("sph/set_acceleration.wgsl"), 
            vec![
                uniform_buffer.as_entire_binding(),
                hashes1.as_entire_binding(),
                cell_range.as_entire_binding(),
                pbuffers.pos.as_entire_binding(),
                pbuffers.vel.as_entire_binding(),
                pbuffers.mass.as_entire_binding(),
                pbuffers.normal.as_entire_binding(),
                pbuffers.density.as_entire_binding(),
                pbuffers.acc.as_entire_binding(),
            ], 
            num_dispatches, 
            Some("set_acceleration"), 
            Some(COMPILATION_OPTIONS),
            Some("set acceleration"), 
            Some("set_acceleration".to_string())
        );

        let move_and_collide = Pipeline::new(
            device, 
            wgpu::include_wgsl!("sph/move_and_collide.wgsl"), 
            vec![
                uniform_buffer.as_entire_binding(),
                pbuffers.pos.as_entire_binding(),
                pbuffers.vel.as_entire_binding(),
                pbuffers.acc.as_entire_binding(),
                instance_buffer.as_entire_binding(),
            ], 
            num_dispatches, 
            Some("move_and_collide"), 
            Some(COMPILATION_OPTIONS),
            Some("move and collide"), 
            Some("move_and_collide".to_string())
        );

        let radix_sort = RadixSort::new(device, &hashes1, &hashes2, n as u32);
        
        Self { pbuffers, hashes1, hashes2, cell_range, instance_buffer, uniform_buffer, calc_hash, find_range, set_density, set_normal, set_acceleration, move_and_collide, radix_sort, n, ubo, control_info: Default::default() }
    }

    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &wgpu_profiler::GpuProfiler,
        control_info: &crate::ControlInfo,
    ) {
        if (control_info.clone() != self.control_info) {
            self.control_info = control_info.clone();
            self.ubo.mouse_x = self.control_info.cursor_pos[0];
            self.ubo.mouse_y = self.control_info.cursor_pos[1];
            self.ubo.mouse_state = (self.control_info.cursor_left_down as u32) + 2 * (self.control_info.cursor_right_down as u32);
            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.ubo]));
        }

        let mut scope = profiler.scope("SPH Compute", encoder);

        //queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.ubo]));
        self.calc_hash.do_pass(&mut scope);
        self.radix_sort.do_sort(&mut scope);
        self.find_range.do_pass(&mut scope);
        self.set_density.do_pass(&mut scope);
        self.set_normal.do_pass(&mut scope);
        self.set_acceleration.do_pass(&mut scope);
        self.move_and_collide.do_pass(&mut scope);
    }
}
