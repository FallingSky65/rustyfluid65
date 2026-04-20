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
@group(0) @binding(1) var<storage, read> hashes: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> cell_range: array<vec2<u32>>;

override WORKGROUP_SIZE: u32 = 256u;

@compute
@workgroup_size(WORKGROUP_SIZE)
fn find_range(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let gid = global_id.x;
    if gid >= ubo.n {
        return;
    }

    if gid == 0 || hashes[gid].x != hashes[gid - 1].x {
        cell_range[hashes[gid].x].x = gid;
    }
    if gid == ubo.n - 1 || hashes[gid].x != hashes[gid + 1].x {
        cell_range[hashes[gid].x].y = gid + 1;
    }
}
