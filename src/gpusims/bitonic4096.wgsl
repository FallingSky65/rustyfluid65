@group(0) @binding(0) var<storage, read_write> particle_hash: array<vec2<u32>>;

var<workgroup> shared_keys: array<u32, 4096>;
var<workgroup> shared_vals: array<u32, 4096>;

fn compare_and_swap(i: u32, j: u32, k: u32) {
    let l = i ^ j;
    if l > i {
        let ascending = (i & k) == 0u;
        if (ascending && shared_keys[i] > shared_keys[l]) ||
            (!ascending && shared_keys[i] < shared_keys[l]) {
            let temp_key = shared_keys[i];
            shared_keys[i] = shared_keys[l];
            shared_keys[l] = temp_key;
            let temp_val = shared_vals[i];
            shared_vals[i] = shared_vals[l];
            shared_vals[l] = temp_val;
        }
    }
}

fn stage(lid: u32, j: u32, k: u32) {
    for (var i = 0u; i < 16u; i++) {
        compare_and_swap(lid * 16u + i, j, k);
    }
}

@compute
@workgroup_size(256)
fn bitonic_sort4096(@builtin(local_invocation_id) lid: vec3<u32>) {
    // load
    for (var i = 0u; i < 16u; i++) {
        let idx = lid.x * 16u + i;
        shared_keys[idx] = particle_hash[idx].x;
        shared_vals[idx] = particle_hash[idx].y;
    }
    workgroupBarrier();

    stage(lid.x, 1u, 2u); workgroupBarrier();

    stage(lid.x, 2u, 4u); workgroupBarrier();
    stage(lid.x, 1u, 4u); workgroupBarrier();

    stage(lid.x, 4u, 8u); workgroupBarrier();
    stage(lid.x, 2u, 8u); workgroupBarrier();
    stage(lid.x, 1u, 8u); workgroupBarrier();

    stage(lid.x, 8u, 16u); workgroupBarrier();
    stage(lid.x, 4u, 16u); workgroupBarrier();
    stage(lid.x, 2u, 16u); workgroupBarrier();
    stage(lid.x, 1u, 16u); workgroupBarrier();

    stage(lid.x, 16u, 32u); workgroupBarrier();
    stage(lid.x, 8u, 32u); workgroupBarrier();
    stage(lid.x, 4u, 32u); workgroupBarrier();
    stage(lid.x, 2u, 32u); workgroupBarrier();
    stage(lid.x, 1u, 32u); workgroupBarrier();

    stage(lid.x, 32u, 64u); workgroupBarrier();
    stage(lid.x, 16u, 64u); workgroupBarrier();
    stage(lid.x, 8u, 64u); workgroupBarrier();
    stage(lid.x, 4u, 64u); workgroupBarrier();
    stage(lid.x, 2u, 64u); workgroupBarrier();
    stage(lid.x, 1u, 64u); workgroupBarrier();

    stage(lid.x, 64u, 128u); workgroupBarrier();
    stage(lid.x, 32u, 128u); workgroupBarrier();
    stage(lid.x, 16u, 128u); workgroupBarrier();
    stage(lid.x, 8u, 128u); workgroupBarrier();
    stage(lid.x, 4u, 128u); workgroupBarrier();
    stage(lid.x, 2u, 128u); workgroupBarrier();
    stage(lid.x, 1u, 128u); workgroupBarrier();

    stage(lid.x, 128u, 256u); workgroupBarrier();
    stage(lid.x, 64u, 256u); workgroupBarrier();
    stage(lid.x, 32u, 256u); workgroupBarrier();
    stage(lid.x, 16u, 256u); workgroupBarrier();
    stage(lid.x, 8u, 256u); workgroupBarrier();
    stage(lid.x, 4u, 256u); workgroupBarrier();
    stage(lid.x, 2u, 256u); workgroupBarrier();
    stage(lid.x, 1u, 256u); workgroupBarrier();

    stage(lid.x, 256u, 512u); workgroupBarrier();
    stage(lid.x, 128u, 512u); workgroupBarrier();
    stage(lid.x, 64u, 512u); workgroupBarrier();
    stage(lid.x, 32u, 512u); workgroupBarrier();
    stage(lid.x, 16u, 512u); workgroupBarrier();
    stage(lid.x, 8u, 512u); workgroupBarrier();
    stage(lid.x, 4u, 512u); workgroupBarrier();
    stage(lid.x, 2u, 512u); workgroupBarrier();
    stage(lid.x, 1u, 512u); workgroupBarrier();

    stage(lid.x, 512u, 1024u); workgroupBarrier();
    stage(lid.x, 256u, 1024u); workgroupBarrier();
    stage(lid.x, 128u, 1024u); workgroupBarrier();
    stage(lid.x, 64u, 1024u); workgroupBarrier();
    stage(lid.x, 32u, 1024u); workgroupBarrier();
    stage(lid.x, 16u, 1024u); workgroupBarrier();
    stage(lid.x, 8u, 1024u); workgroupBarrier();
    stage(lid.x, 4u, 1024u); workgroupBarrier();
    stage(lid.x, 2u, 1024u); workgroupBarrier();
    stage(lid.x, 1u, 1024u); workgroupBarrier();

    stage(lid.x, 1024u, 2048u); workgroupBarrier();
    stage(lid.x, 512u, 2048u); workgroupBarrier();
    stage(lid.x, 256u, 2048u); workgroupBarrier();
    stage(lid.x, 128u, 2048u); workgroupBarrier();
    stage(lid.x, 64u, 2048u); workgroupBarrier();
    stage(lid.x, 32u, 2048u); workgroupBarrier();
    stage(lid.x, 16u, 2048u); workgroupBarrier();
    stage(lid.x, 8u, 2048u); workgroupBarrier();
    stage(lid.x, 4u, 2048u); workgroupBarrier();
    stage(lid.x, 2u, 2048u); workgroupBarrier();
    stage(lid.x, 1u, 2048u); workgroupBarrier();

    stage(lid.x, 2048u, 4096u); workgroupBarrier();
    stage(lid.x, 1024u, 4096u); workgroupBarrier();
    stage(lid.x, 512u, 4096u); workgroupBarrier();
    stage(lid.x, 256u, 4096u); workgroupBarrier();
    stage(lid.x, 128u, 4096u); workgroupBarrier();
    stage(lid.x, 64u, 4096u); workgroupBarrier();
    stage(lid.x, 32u, 4096u); workgroupBarrier();
    stage(lid.x, 16u, 4096u); workgroupBarrier();
    stage(lid.x, 8u, 4096u); workgroupBarrier();
    stage(lid.x, 4u, 4096u); workgroupBarrier();
    stage(lid.x, 2u, 4096u); workgroupBarrier();
    stage(lid.x, 1u, 4096u); workgroupBarrier();

    // store
    for (var i = 0u; i < 16u; i++) {
        let idx = lid.x * 16u + i;
        particle_hash[idx].x = shared_keys[idx];
        particle_hash[idx].y = shared_vals[idx];
    }
}
