@group(0) @binding(0) var<storage, read_write> global_prefix: array<u32>;
@group(0) @binding(1) var<storage, read_write> aux: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
struct UBO {
    n: u32,
    shift: u32,
    num_blocks: u32, // n.ceildiv(256)
    pingpong: u32,
};
@group(1) @binding(0) var<uniform> ubo: UBO;

var<workgroup> s_tile: array<u32, 256u>;

fn scan_tile(lid: u32) -> u32 {
    // upsweep
    for (var stride: u32 = 1u; stride < 256u; stride <<= 1u) {
        let wi = (lid + 1u) * stride * 2u - 1u;
        let ri = wi - stride;
        if wi < 256u {
            s_tile[wi] += s_tile[ri];
        }
        workgroupBarrier();
    }

    var total = 0u;
    if lid == 0u {
        total = s_tile[255u];
        s_tile[255u] = 0;
    }
    workgroupBarrier();

    // downsweep
    for (var stride: u32 = 128u; stride > 0; stride >>= 1u) {
        let wi = (lid + 1u) * stride * 2u - 1u;
        let ri = wi - stride;
        if wi < 256u {
            let t = s_tile[ri];
            s_tile[ri] = s_tile[wi];
            s_tile[wi] += t;
        }
        workgroupBarrier();
    }

    return total;
}

@compute
@workgroup_size(256)
fn reduce(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let lid = local_id.x;
    let total_entries = ubo.num_blocks * 4u;
    let tile_base = workgroup_id.x * 256u;
    let gid = tile_base + lid;

    s_tile[lid] = select(0u, block_sums[gid], gid < total_entries);
    workgroupBarrier();

    let tile_total = scan_tile(lid);

    if gid < total_entries {
        global_prefix[gid] = s_tile[lid];
    }

    if lid == 0u {
        aux[workgroup_id.x] = tile_total;
    }
}

@compute
@workgroup_size(256)
fn downsweep(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let lid = local_id.x;
    let num_tiles = (ubo.num_blocks * 4u + 255u) / 256u;
    s_tile[lid] = select(0u, aux[lid], lid < num_tiles);
    workgroupBarrier();

    scan_tile(lid);

    let total_entries = ubo.num_blocks * 4u;
    let tile_offset = s_tile[lid];
    for (var j: u32 = lid * 256u; j < min(lid * 256u + 256u, total_entries); j++) {
        global_prefix[j] += tile_offset;
    }
}
