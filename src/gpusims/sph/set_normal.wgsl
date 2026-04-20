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
@group(0) @binding(2) var<storage, read> cell_range: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> pos: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> mass: array<f32>;
@group(0) @binding(5) var<storage, read> density: array<f32>;
@group(0) @binding(6) var<storage, read_write> normal: array<vec2<f32>>;

override WORKGROUP_SIZE: u32 = 256u;

const neighbors: array<vec2<i32>, 9> = array(
    vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1),
    vec2<i32>(-1, 0), vec2<i32>(0, 0), vec2<i32>(1, 0),
vec2<i32>(-1, 1), vec2<i32>(0, 1), vec2<i32>(1, 1),
);

fn hash2D(v: vec2<i32>) -> u32 {
    return bitcast<u32>(v.x * 18397 + v.y * 20483);
}

const pi: f32 = 3.141592653589793238462643;

fn W_cubic_deriv(r: f32, h: f32) -> f32 {
    let q = abs(r / h);
    const factor2d = 40.0 / (7 * pi);
    return factor2d / pow(h, 3) * select(
        0.0,
        select(
            -12 * q + 18 * q * q,
            -6 * pow(1 - q, 2),
            q > 0.5
        ),
        q <= 1
    );
}

@compute
@workgroup_size(WORKGROUP_SIZE)
fn set_normal(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let gid = global_id.x;
    if gid >= ubo.n {
        return;
    }

    var normal1 = vec2<f32>(0.0);
    let pos1 = pos[gid];
    let cell_pos = vec2<i32>(floor(pos1 / ubo.h));

    for (var i: u32 = 0; i < 9u; i += 1) {
        let cell: vec2<i32> = cell_pos + neighbors[i];
        let range: vec2<u32> = cell_range[hash2D(cell) % ubo.n];
        for (var j: u32 = range.x; j < range.y; j += 1) {
            let k = hashes[j].y;
            let pos2 = pos[k];
            let mass2 = mass[k];
            let density2 = density[k];
            if length(pos2 - pos1) > 0 {
                normal1 += ubo.h * mass2 / density2 * W_cubic_deriv(length(pos2 - pos1), ubo.h) * normalize(pos2 - pos1);
            }
        }
    }
    
    normal[gid] = normal1;
}
