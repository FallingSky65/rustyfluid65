struct UBO {
    n: u32, // number of particles
    h: f32, // smoothing radius
    gas_constant: f32,
    target_density: f32,
    canvas_width: f32,
    canvas_height: f32,
    dt: f32,
    mouse_x: f32,
    mouse_y: f32,
    mouse_state: u32,
};
@group(0) @binding(0) var<uniform> ubo: UBO;
@group(0) @binding(1) var<storage, read> hashes: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> cell_range: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> pos: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> mass: array<f32>;
@group(0) @binding(5) var<storage, read_write> density: array<f32>;

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

const boxR: f32 = 10.0;

fn W_poly6(r: f32, h: f32) -> f32 {
    const factor2d = 4.0 / pi;
    return factor2d / pow(h, 8) * select(
        0.0,
        pow(h * h - r * r, 3),
        r <= h
    );
}

@compute
@workgroup_size(WORKGROUP_SIZE)
fn set_density(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let gid = global_id.x;
    if gid >= ubo.n {
        return;
    }
    
    var density1: f32 = 0.0;
    let pos1 = pos[gid];
    let cell_pos = vec2<i32>(floor(pos1 / ubo.h));

    for (var i: u32 = 0; i < 9u; i += 1) {
        let cell: vec2<i32> = cell_pos + neighbors[i];
        let range: vec2<u32> = cell_range[hash2D(cell) % ubo.n];
        for (var j: u32 = range.x; j < range.y; j += 1) {
            let pos2 = pos[hashes[j].y];
            let mass2 = mass[hashes[j].y];
            density1 += mass2 * W_poly6(length(pos2 - pos1), ubo.h);

            var pos3 = pos2;
            if pos1.x < ubo.h - boxR {
                pos3.x = -2.0 * boxR - pos2.x;
                density1 += mass2 * W_poly6(length(pos3 - pos1), ubo.h);

                if pos1.y < ubo.h - boxR {
                    pos3.y = -2.0 * boxR - pos2.y;
                    density1 += mass2 * W_poly6(length(pos3 - pos1), ubo.h);
                }
            } else if pos1.x > boxR - ubo.h {
                pos3.x = 2.0 * boxR - pos2.x;
                density1 += mass2 * W_poly6(length(pos3 - pos1), ubo.h);

                if pos1.y < ubo.h - boxR {
                    pos3.y = -2.0 * boxR - pos2.y;
                    density1 += mass2 * W_poly6(length(pos3 - pos1), ubo.h);
                }
            } else if pos1.y < ubo.h - boxR {
                pos3.y = -2.0 * boxR - pos2.y;
                density1 += mass2 * W_poly6(length(pos3 - pos1), ubo.h);
            }
        }
    }

    density[gid] = density1;
}
