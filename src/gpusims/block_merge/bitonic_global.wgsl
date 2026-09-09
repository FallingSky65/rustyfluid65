@group(0) @binding(0) var<storage, read_write> arr: array<vec2<u32>>;

struct UBO {
    n: u32,
    block_size: u32,
    flip: u32,
};
@group(1) @binding(0) var<uniform> ubo: UBO;


@compute
@workgroup_size(256)
fn bitonic_global(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let gid = global_id.x;      // global id
    let lid = local_id.x;       // local id
    let wid = workgroup_id.x;   // workgroup id

    let block_size = ubo.block_size;
    let block_index = gid / (block_size/2);
    let pos_in_block = gid % (block_size/2);

    let a = block_index * block_size + pos_in_block;
    let b = select(
        a + block_size/2,
        (block_index + 1) * block_size - pos_in_block - 1,
        ubo.flip == 1
    );

    if ((a >= ubo.n) || (b >= ubo.n)) {
        return;
    }

    if (arr[a].x > arr[b].x) {
        let temp = arr[a];
        arr[a] = arr[b];
        arr[b] = temp;
    }
}
