mod gpusims;
mod simulations;

use std::{sync::Arc, time::Duration};

use wgpu::util::DeviceExt;
use wgpu_profiler::GpuProfilerSettings;
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[allow(unused_imports)]
use crate::gpusims::{
    GPUSimulation, rotate::GPURotateSim, sph::GPUSmoothedParticleHydrodynamicsSim,
};
#[allow(unused_imports)]
use crate::simulations::{
    Simulation, rigidbody::RigidBodySim, rotate::RotateSim, sph::SmoothedParticleHydrodynamicsSim,
};

const CANVAS_SIZE: [u32; 2] = [512, 512];
const CANVAS_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    canvas_size: [f32; 2],
    res: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        }
    }
}

const QUAD_VERTICES: &[Vertex] = &[
    Vertex {
        position: [-1.0, -1.0],
    },
    Vertex {
        position: [1.0, -1.0],
    },
    Vertex {
        position: [-1.0, 1.0],
    },
    Vertex {
        position: [1.0, 1.0],
    },
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    position: [f32; 4],
    color: [f32; 4],
}

impl Instance {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Instance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

const FRAMES_IN_FLIGHT: usize = 2;

#[allow(unused)]
pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    profiler: wgpu_profiler::GpuProfiler,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Window>,

    render_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,

    uniform: Uniform,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    // instances: Vec<Instance>,
    // instance_buffer: wgpu::Buffer,
    canvas: wgpu::Texture,
    canvas_view: wgpu::TextureView,
    canvas_bind_group: wgpu::BindGroup,

    latest_profiler_results: Option<Vec<wgpu_profiler::GpuTimerQueryResult>>,
    frame_index: usize,

    // sim: Box<dyn Simulation>,
    gpusim: Box<dyn GPUSimulation>,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<State> {
        let size = window.inner_size();

        println!("size: {} x {}", size.width, size.height);

        // TODO
        // may need correct backend for web
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        // instance_desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(instance_desc);

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::downlevel_defaults()
                },
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await?;

        let profiler = wgpu_profiler::GpuProfiler::new(&device, GpuProfilerSettings::default())
            .expect("Failed to create profiler");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync, // surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: FRAMES_IN_FLIGHT as u32,
        };
        surface.configure(&device, &config);

        let uniform = Uniform {
            canvas_size: [CANVAS_SIZE[0] as f32, CANVAS_SIZE[1] as f32],
            res: [size.width as f32, size.height as f32],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let canvas = device.create_texture(&wgpu::wgt::TextureDescriptor {
            label: Some("canvas"),
            size: wgpu::Extent3d {
                width: CANVAS_SIZE[0],
                height: CANVAS_SIZE[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CANVAS_FMT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let canvas_view = canvas.create_view(&wgpu::TextureViewDescriptor::default());

        let canvas_sampler = device.create_sampler(&wgpu::wgt::SamplerDescriptor {
            label: Some("canvas_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let canvas_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("canvas_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let canvas_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canvas_bind_group"),
            layout: &canvas_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&canvas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&canvas_sampler),
                },
            ],
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[Some(&uniform_bind_group_layout)],
                ..Default::default()
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), Instance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: CANVAS_FMT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Pipeline Layout"),
            bind_group_layouts: &[Some(&canvas_bind_group_layout)],
            ..Default::default()
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        //let sim = Box::new(RotateSim::new(1000, 1.0));
        // let sim = Box::new(RigidBodySim::new(
        //     1000,
        //     [CANVAS_SIZE[0] as f32, CANVAS_SIZE[1] as f32],
        //     10.0,
        // ));

        // let instances = sim.get_instances();

        // let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("Instance Buffer"),
        //     contents: bytemuck::cast_slice(instances.as_slice()),
        //     usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        // });

        #[allow(non_snake_case)]
        let simUBO = gpusims::sph::UBO {
            nparticles: 1024,
            smooth_radius: 1.0,
            block_size: 0,
            flip: 0,
            gas_constant: 100.0,
            target_density: 3.6,
            canvas_width: CANVAS_SIZE[0] as f32,
            canvas_height: CANVAS_SIZE[1] as f32,
            dt: 0.005,
        };
        let gpusim = Box::new(GPUSmoothedParticleHydrodynamicsSim::new(
            &device, 100.0, 10.0, simUBO,
        ));

        Ok(Self {
            surface,
            device,
            queue,
            profiler,
            config,
            is_surface_configured: false,
            window,

            render_pipeline,
            blit_pipeline,
            vertex_buffer,

            uniform,
            uniform_buffer,
            uniform_bind_group,

            // instances,
            // instance_buffer,
            canvas,
            canvas_view,
            canvas_bind_group,

            latest_profiler_results: None,
            frame_index: 1,

            // sim,
            gpusim,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;

            self.uniform.res[0] = width as f32;
            self.uniform.res[1] = height as f32;
            self.queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(&[self.uniform]),
            );
        }
    }

    fn handle_key(&self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {}
        }
    }

    pub fn draw_to_canvas(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.canvas_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            multiview_mask: None,
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        //render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        //render_pass.draw(0..4, 0..self.instances.len() as _);
        render_pass.set_vertex_buffer(1, self.gpusim.get_instances_slice());
        render_pass.draw(0..4, 0..self.gpusim.get_instances_len() as _);
    }

    pub fn render(&mut self) {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return;
        }

        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => {
                let view = surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Render Encoder"),
                        });

                self.draw_to_canvas(&mut encoder);

                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        multiview_mask: None,
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });

                    render_pass.set_pipeline(&self.blit_pipeline);
                    render_pass.set_bind_group(0, &self.canvas_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

                    if self.uniform.res[0] > self.uniform.res[1] {
                        render_pass.set_viewport(
                            (self.uniform.res[0] - self.uniform.res[1]) * 0.5,
                            0.0,
                            self.uniform.res[1],
                            self.uniform.res[1],
                            0.0,
                            1.0,
                        );
                    } else if self.uniform.res[0] < self.uniform.res[1] {
                        render_pass.set_viewport(
                            0.0,
                            (self.uniform.res[1] - self.uniform.res[0]) * 0.5,
                            self.uniform.res[0],
                            self.uniform.res[0],
                            0.0,
                            1.0,
                        );
                    }

                    render_pass.draw(0..4, 0..1);
                }

                self.queue.submit(std::iter::once(encoder.finish()));
                surface_texture.present();
            }
            wgpu::CurrentSurfaceTexture::Suboptimal(_)
            | wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Outdated => {
                log::error!("Unable to render: Suboptimal/Lost/Outdated");
                let size = self.window.inner_size();
                self.resize(size.width, size.height);
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                log::error!("Unable to render: Occluded");
            }
            wgpu::CurrentSurfaceTexture::Timeout => {
                log::error!("Unable to render: Timeout");
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                log::error!("Unable to render: Timeout");
            }
        }
    }

    fn update(&mut self) -> (Duration, Vec<Duration>) {
        let cpu_start = std::time::Instant::now();
        // self.sim.sim_update(1.0 / 60.0);
        profiling::scope!("Simulation Update");
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.gpusim
            .sim_update(&self.queue, &mut encoder, &self.profiler);
        self.profiler.resolve_queries(&mut encoder);

        {
            profiling::scope!("Submit");
            self.queue.submit(Some(encoder.finish()));
        }

        profiling::finish_frame!();

        self.profiler.end_frame().unwrap();

        self.latest_profiler_results = self
            .profiler
            .process_finished_frame(self.queue.get_timestamp_period());

        print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
        let gpu_time = self.print_results(1);

        self.frame_index += 1;

        // self.queue.write_buffer(
        // &self.instance_buffer,
        // 0,
        // bytemuck::cast_slice(self.sim.get_instances().as_slice()),
        // );

        let cpu_time = cpu_start.elapsed();

        (cpu_time, gpu_time)
    }

    fn recurse_results(
        &self,
        results: &[wgpu_profiler::GpuTimerQueryResult],
        depth: usize,
        result_vec: &mut Vec<(f64, String, usize)>,
    ) -> f64 {
        let mut accumulate: f64 = 0.0;

        for scope in results {
            let i = result_vec.len();
            result_vec.push((0.0, scope.label.clone(), depth));

            let mut scope_time = match &scope.time {
                Some(time) => time.end - time.start,
                None => 0.0,
            };

            if !scope.nested_queries.is_empty() {
                scope_time += self.recurse_results(&scope.nested_queries, depth + 1, result_vec);
            }

            if let Some(result) = result_vec.get_mut(i) {
                result.0 = scope_time;
            }

            accumulate += scope_time;
        }

        accumulate
    }

    fn print_results(&self, depth: usize) -> Vec<Duration> {
        let mut durations: Vec<Duration> = Vec::new();

        if let Some(results) = self.latest_profiler_results.as_ref() {
            let mut result_vec: Vec<(f64, String, usize)> = Vec::new();

            self.recurse_results(results.as_slice(), 0, &mut result_vec);

            for result in result_vec {
                if result.0 > 0.0 {
                    if result.2 <= depth {
                        if result.2 > 0 {
                            print!("{:<width$}", "|", width = 4 * result.2);
                        } else {
                            durations.push(Duration::from_secs_f64(result.0));
                        }
                        println!("{:.3}µs - {}", result.0 * 1000.0 * 1000.0, result.1,)
                    }
                }
            }
        }

        durations
    }
}

struct AppStats {
    cpu_times: Vec<Duration>,
    gpu_times: Vec<Duration>,
    frame_times: Vec<Duration>,
    index: usize,
    gpu_index: usize,
    count: usize,

    last_frame: std::time::Instant,
}

#[allow(unused)]
impl AppStats {
    fn new(history: usize) -> Self {
        Self {
            cpu_times: vec![Duration::ZERO; history],
            gpu_times: vec![Duration::ZERO; history],
            frame_times: vec![Duration::ZERO; history],
            index: 0,
            gpu_index: 0,
            count: 0,
            last_frame: std::time::Instant::now(),
        }
    }

    fn push(&mut self, cpu: Duration, gpu: Vec<Duration>) {
        let frame_time = self.last_frame.elapsed();
        self.last_frame = std::time::Instant::now();

        self.cpu_times[self.index] = cpu;
        for gpu_time in gpu {
            self.gpu_times[self.gpu_index] = gpu_time;
            self.gpu_index = (self.gpu_index + 1) % self.gpu_times.len();
        }
        self.frame_times[self.index] = frame_time;

        self.index = (self.index + 1) % self.cpu_times.len();
        self.count = self.count.max(self.index).min(self.cpu_times.len());
    }

    fn avg_cpu(&self) -> Duration {
        self.avg(&self.cpu_times)
    }

    fn avg_gpu(&self) -> Duration {
        self.avg(&self.gpu_times)
    }

    fn avg_frame(&self) -> Duration {
        self.avg(&self.frame_times)
    }

    fn avg_fps(&self) -> f64 {
        let avg = self.avg(&self.frame_times);
        if avg.is_zero() {
            0.0
        } else {
            1.0 / avg.as_secs_f64()
        }
    }

    fn avg(&self, buf: &[Duration]) -> Duration {
        let filled = &buf[..self.count];
        if filled.is_empty() {
            return Duration::ZERO;
        }
        filled.iter().sum::<Duration>() / filled.len() as u32
    }
}

pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
    stats: AppStats,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());

        let stats = AppStats::new(128);

        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
            stats,
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(
                        proxy
                            .send_event(State::new(window).await.expect("Unable to create canvas"))
                            .is_ok()
                    )
                });
            }
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, mut event: State) {
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );
        }
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                let (cpu_time, gpu_time) = state.update();

                state.render();

                self.stats.push(cpu_time, gpu_time);

                if self.stats.index % 32 == 0 {
                    state.window.set_title(&format!(
                        "FPS: {:.2}   CPU: {:.4?}   GPU: {:.4?}   Frame: {:.4?}",
                        self.stats.avg_fps(),
                        self.stats.avg_cpu(),
                        self.stats.avg_gpu(),
                        self.stats.avg_frame(),
                    ));
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };
        state.window.request_redraw();
    }
}

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
