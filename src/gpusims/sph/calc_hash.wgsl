struct UBO {
    n: u32, // number of particles
    h: f32, // smoothing radius
    gas_constant: f32,
    target_density: f32,
    canvas_width: f32,
    canvas_height: f32,
    dt: f32,
};
@group(0) @binding(0) var<uniform> ubo: UBO;
@group(0) @binding(1) var<storage, read> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> hashes: array<vec2<u32>>;

override WORKGROUP_SIZE: u32 = 256u;

fn hash2D(v: vec2<i32>) -> u32 {
    return bitcast<u32>(v.x * 18397 + v.y * 20483);
}

@compute
@workgroup_size(WORKGROUP_SIZE)
fn calc_hash(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let gid = global_id.x;
    if gid >= ubo.n {
        return;
    }

    hashes[gid] = vec2<u32>(
        hash2D(vec2<i32>(floor(pos[gid] / ubo.h))) % ubo.n, gid
    );
}
