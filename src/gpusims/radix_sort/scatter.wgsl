@group(0) @binding(0) var<storage, read_write> hashes1: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> hashes2: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> local_prefix: array<u32>;
@group(0) @binding(3) var<storage, read_write> global_prefix: array<u32>;
struct UBO {
    n: u32,
    shift: u32,
    num_blocks: u32, // n.ceildiv(256)
    pingpong: u32,
};
@group(1) @binding(0) var<uniform> ubo: UBO;

@compute
@workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let gid = global_id.x;
    if gid >= ubo.n {
        return;
    }
    let bid = workgroup_id.x;

    let in = select(hashes2[gid], hashes1[gid], ubo.pingpong == 0);
    let digit = (in.x >> ubo.shift) & 3u;
    //let block_id = gid / 256u;
    let pos = global_prefix[digit * ubo.num_blocks + bid] + local_prefix[gid];
    if ubo.pingpong == 0 {
        hashes2[pos] = in;
    } else {
        hashes1[pos] = in;
    }
}
