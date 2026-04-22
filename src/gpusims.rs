use std::any::Any;

pub mod rotate;
pub mod sph;

pub trait GPUSimulation {
    fn get_instances_slice(&self) -> wgpu::BufferSlice<'_>;
    fn get_instances_len(&self) -> usize;
    fn sim_update(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &wgpu_profiler::GpuProfiler,
        control_info: &crate::ControlInfo,
    );
    #[allow(unused)]
    fn as_any(&self) -> &dyn Any;
}

impl GPUSimulation for rotate::GPURotateSim {
    fn get_instances_len(&self) -> usize {
        return self.n;
    }

    fn get_instances_slice(&self) -> wgpu::BufferSlice<'_> {
        return self.instance_buffer.slice(..);
    }

    fn sim_update(
        &mut self,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _profiler: &wgpu_profiler::GpuProfiler,
        _control_info: &crate::ControlInfo,
    ) {
        // self.update(device, queue);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl GPUSimulation for sph::GPUSmoothedParticleHydrodynamicsSim {
    fn get_instances_len(&self) -> usize {
        return self.n;
    }

    fn get_instances_slice(&self) -> wgpu::BufferSlice<'_> {
        return self.instance_buffer.slice(..);
    }

    fn sim_update(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &wgpu_profiler::GpuProfiler,
        control_info: &crate::ControlInfo,
    ) {
        self.update(queue, encoder, profiler, control_info);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
