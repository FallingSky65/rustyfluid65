@group(0) @binding(0) var<storage, read_write> hashes1: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> hashes2: array<vec2<u32>>;

// block_sums[digit * num_blocks + block_id] = count of digit in block
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

// local_prefix[i] = rank of i in (block, digit) group
@group(0) @binding(3) var<storage, read_write> local_prefix: array<u32>;

struct UBO {
    n: u32,
    shift: u32,
    num_blocks: u32, // n.ceildiv(256)
    pingpong: u32,
};
@group(1) @binding(0) var<uniform> ubo: UBO;

var<workgroup> s_data: array<vec2<u32>, 256>;
var<workgroup> s_count: array<array<u32, 256>, 4>;
var<workgroup> s_shuf: array<vec2<u32>, 256>;

@compute
@workgroup_size(128)
fn local_scan(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let gid = global_id.x;
    let lid = local_id.x;
    let bid = workgroup_id.x; // block_id
    let block_base = bid * 256u;

    let i0 = block_base + lid;
    let i1 = i0 + 128u;

    // load into workgroup shared memory
    if ubo.pingpong == 0 {
        s_data[lid] = select(vec2<u32>(0xFFFFFFFFu), hashes1[i0], i0 < ubo.n);
        s_data[lid + 128u] = select(vec2<u32>(0xFFFFFFFFu), hashes1[i1], i1 < ubo.n);
    } else {
        s_data[lid] = select(vec2<u32>(0xFFFFFFFFu), hashes2[i0], i0 < ubo.n);
        s_data[lid + 128u] = select(vec2<u32>(0xFFFFFFFFu), hashes2[i1], i1 < ubo.n);
    }
    workgroupBarrier();

    // extract 2-bit digit
    let digit0 = (s_data[lid].x >> ubo.shift) & 3u;
    let digit1 = (s_data[lid + 128u].x >> ubo.shift) & 3u;
    // count
    for (var d: u32 = 0u; d < 4u; d++) {
        s_count[d][lid] = select(0u, 1u, digit0 == d && i0 < ubo.n);
        s_count[d][lid + 128u] = select(0u, 1u, digit1 == d && i1 < ubo.n);
    }
    workgroupBarrier();

    // prefix sums on all 4 bitmasks
    // upsweep
    for (var stride: u32 = 1u; stride < 256u; stride <<= 1u) {
        let write_idx = (lid + 1u) * stride * 2u - 1u;
        let read_idx = write_idx - stride;
        if write_idx < 256u {
            for (var d: u32 = 0u; d < 4u; d++) {
                s_count[d][write_idx] += s_count[d][read_idx];
            }
        }
        workgroupBarrier();
    }

    // total sum is now in s_count[d][255]
    // save into block_sums
    if lid == 0u {
        for (var d: u32 = 0u; d < 4u; d++) {
            block_sums[d * ubo.num_blocks + bid] = s_count[d][255u];
            s_count[d][255u] = 0u; // zeroed for exclusive scan
        }
    }
    workgroupBarrier();

    // downsweep
    for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
        let write_idx = (lid + 1u) * stride * 2u - 1u;
        let read_idx = write_idx - stride;
        if write_idx < 256u {
            for (var d: u32 = 0u; d < 4u; d++) {
                let temp = s_count[d][read_idx];
                s_count[d][read_idx] = s_count[d][write_idx];
                s_count[d][write_idx] += temp;
            }
        }
        workgroupBarrier();
    }
    
    // write local prefix ranks to local_prefix buffer
    if i0 < ubo.n {
        local_prefix[i0] = s_count[digit0][lid];
    }
    if i1 < ubo.n {
        local_prefix[i1] = s_count[digit1][lid + 128u];
    }

    return;

    workgroupBarrier();

    // in block shuffle
    let total0 = block_sums[0u * ubo.num_blocks + bid];
    let total1 = block_sums[1u * ubo.num_blocks + bid];
    let total2 = block_sums[2u * ubo.num_blocks + bid];
    let off0 = 0u;
    let off1 = total0;
    let off2 = total0 + total1;
    let off3 = total0 + total1 + total2;

    let shuf0 = select(
        select(
            off3 + s_count[3][lid],
            off2 + s_count[2][lid],
            digit0 == 2u,
        ),
        select(
            off1 + s_count[1][lid],
            off0 + s_count[0][lid],
            digit0 == 0u,
        ),
        digit0 < 2u
    );
    let shuf1 = select(
        select(
            off3 + s_count[3][lid + 128u],
            off2 + s_count[2][lid + 128u],
            digit1 == 2u,
        ),
        select(
            off1 + s_count[1][lid + 128u],
            off0 + s_count[0][lid + 128u],
            digit1 == 0u,
        ),
        digit1 < 2u
    );
    

    s_shuf[shuf0] = s_data[lid];
    s_shuf[shuf1] = s_data[lid + 128u];
    workgroupBarrier();

    // scatter: write back to global
    if i0 < ubo.n {
        hashes2[i0] = s_shuf[lid];
    }
    if i1 < ubo.n {
        hashes2[i1] = s_shuf[lid + 128u];
    }
}
