@group(0) @binding(0) var<storage, read_write> arr: array<vec2<u32>>;

struct UBO {
    n: u32,
};
@group(1) @binding(0) var<uniform> ubo: UBO;

var<workgroup> w_data: array<vec2<u32>, 2048>;

// compare and swap
fn cas(
    k: ptr<function, array<u32, 8>>,
    v: ptr<function, array<u32, 8>>,
    i: u32,
    j: u32,
    dir: bool,
) {
    let k_i = (*k)[i];
    let k_j = (*k)[j];
    let v_i = (*v)[i];
    let v_j = (*v)[j];

    let swap = (dir && k_i > k_j) || (!dir && k_i < k_j);

    (*k)[i] = select(k_i, k_j, swap);
    (*k)[j] = select(k_j, k_i, swap);
    (*v)[i] = select(v_i, v_j, swap);
    (*v)[j] = select(v_j, v_i, swap);
}

@compute
@workgroup_size(256)
fn bitonic_local(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let gid = global_id.x;      // global id
    let lid = local_id.x;       // local id
    let wid = workgroup_id.x;   // workgroup id

    var f_keys: array<u32, 8>;
    var f_vals: array<u32, 8>;

    // load into function memory
    for (var i: u32 = 0u; i < 8u; i++) {
        if wid*2048 + lid*8u + i < ubo.n {
            f_keys[i] = arr[wid*2048u + lid*8u + i].x;
            f_vals[i] = arr[wid*2048u + lid*8u + i].y;
        } else {
            f_keys[i] = 0xFFFFFFFFu;
            f_vals[i] = 0xFFFFFFFFu;
        }
    }
    
    // sort 8-element chunk in function memory
    let dir = (lid & 1u) == 0u;
    cas(&f_keys, &f_vals, 0u, 1u, dir);
    cas(&f_keys, &f_vals, 2u, 3u, dir);
    cas(&f_keys, &f_vals, 4u, 5u, dir);
    cas(&f_keys, &f_vals, 6u, 7u, dir);
    
    cas(&f_keys, &f_vals, 0u, 2u, dir);
    cas(&f_keys, &f_vals, 1u, 3u, dir);
    cas(&f_keys, &f_vals, 4u, 6u, dir);
    cas(&f_keys, &f_vals, 5u, 7u, dir);
    
    cas(&f_keys, &f_vals, 1u, 2u, dir);
    cas(&f_keys, &f_vals, 5u, 6u, dir);
    cas(&f_keys, &f_vals, 0u, 4u, dir);
    cas(&f_keys, &f_vals, 3u, 7u, dir);
    
    cas(&f_keys, &f_vals, 1u, 5u, dir);
    cas(&f_keys, &f_vals, 2u, 6u, dir);
    
    cas(&f_keys, &f_vals, 1u, 4u, dir);
    cas(&f_keys, &f_vals, 3u, 6u, dir);
    
    cas(&f_keys, &f_vals, 2u, 4u, dir);
    cas(&f_keys, &f_vals, 3u, 5u, dir);
    
    cas(&f_keys, &f_vals, 3u, 4u, dir);

    // load into workgroup shared memory
    for (var i: u32 = 0u; i < 8u; i++) {
        w_data[lid*8u + i].x = f_keys[i];
        w_data[lid*8u + i].y = f_vals[i];
    }
    workgroupBarrier();

    // bitonic merge network
    for (var block_size: u32 = 16u; block_size <= 2048u; block_size <<= 1u) {
        for (var stride: u32 = block_size >> 1u; stride > 0u; stride >>= 1u) {
            for (var i = 0u; i < 4u; i++) {
                let pid = lid + i*256u; // pair id
                
                let left = ((pid/stride)*(stride*2u)) + (pid % stride);
                let right = left + stride;
                let dir = ((left/block_size) & 1u) == 0u;
                
                if ((dir && w_data[left].x > w_data[right].x) || (!dir && w_data[left].x < w_data[right].x)) {
                    let temp: vec2<u32> = w_data[left];
                    w_data[left] = w_data[right];
                    w_data[right] = temp;
                }
            }
            workgroupBarrier();
        }
    }

    // store into device memory
    for (var i: u32 = 0u; i < 8u; i++) {
        if wid*2048 + lid*8u + i < ubo.n {
            arr[wid*2048 + lid*8u + i] = w_data[lid*8u + i];
        }
    }
}
