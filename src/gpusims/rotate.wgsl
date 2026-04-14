struct particle {
    position: vec2<f32>,
};

struct instance {
    position: vec4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> particles: array<particle>;
@group(0) @binding(1) var<storage, read_write> instances: array<instance>;

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if gid.x >= arrayLength(&particles) {
        return;
    }

    let s = sin(0.1);
    let c = cos(0.1);

    particles[gid.x].position = c * particles[gid.x].position + vec2(-s, s) * particles[gid.x].position.yx;

    instances[gid.x].position = 5 * vec4(particles[gid.x].position * 50 + 50, 0.0, 0.0);
    instances[gid.x].color = vec4(1, 1, 1, 0);
}
