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
@group(0) @binding(4) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> mass: array<f32>;
@group(0) @binding(6) var<storage, read> normal: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read> density: array<f32>;
@group(0) @binding(8) var<storage, read_write> acc: array<vec2<f32>>;

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

fn W_spiky_grad(r_vec: vec2<f32>, h: f32) -> vec2<f32> {
    let r = length(r_vec);
    if r == 0.0 {
        return vec2<f32>(0.0);
    }
    const factor2d = -30.0 / pi;
    return select(
        vec2<f32>(0.0),
        factor2d / pow(h, 5) * pow(h - r, 2) * normalize(r_vec),
        r <= h
    );
}

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

fn W_cohesion(r: f32, h: f32) -> f32 {
    const factor2d: f32 = 80.0 / (3.0 * pi);
    return factor2d / pow(h, 8) * select(
        0.0,
        select(
            2 * pow((h - r) * r, 3) - pow(0.5 * h, 6),
            pow((h - r) * r, 3),
            r > 0.5
        ),
        r <= h
    );
}

@compute
@workgroup_size(WORKGROUP_SIZE)
fn set_acceleration(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let gid = global_id.x;
    if gid >= ubo.n {
        return;
    }

    var F_pressure = vec2<f32>(0.0);
    var F_viscosity = vec2<f32>(0.0);
    var F_cohestion = vec2<f32>(0.0);
    var F_curvature = vec2<f32>(0.0);
    let pos1 = pos[gid];
    let vel1 = vel[gid];
    let mass1 = mass[gid];
    let normal1 = normal[gid];
    let density1 = density[gid];
    let pressure1 = max(ubo.gas_constant * (pow(density1 / ubo.target_density, 7) - 1), 0);
    let cell_pos = vec2<i32>(floor(pos1 / ubo.h));
    
    for (var i: u32 = 0; i < 9u; i += 1) {
        let cell: vec2<i32> = cell_pos + neighbors[i];
        let range: vec2<u32> = cell_range[hash2D(cell) % ubo.n];
        for (var j: u32 = range.x; j < range.y; j += 1) {
            let k = hashes[j].y;
            let pos2 = pos[k];
            let vel2 = vel[k];
            let mass2 = mass[k];
            let normal2 = normal[k];
            let density2 = density[k];
            let pressure2 = max(ubo.gas_constant * (pow(density2 / ubo.target_density, 7) - 1), 0);
            let dst = length(pos2 - pos1);

            F_pressure += W_spiky_grad(pos2 - pos1, ubo.h) * (pressure2 / (density2 * density2) + pressure1 / (density1 * density1)) * 0.5 * mass2;

            if dst > 0.01 {
                F_viscosity += (mass2 / density2) * 2 * (vel1 - vel2) / dst * W_cubic_deriv(dst, ubo.h);
                F_cohestion += mass1 * W_cohesion(dst, ubo.h) * normalize(pos1 - pos2);
                F_curvature += mass1 * select(
                    vec2<f32>(0.0),
                    normal1 - normal2,
                    dst < ubo.h,
                );
            }
            
            var pos3 = pos2;
            if pos1.x < ubo.h - boxR {
                pos3.x = -2.0 * boxR - pos2.x;
                F_pressure += W_spiky_grad(pos3 - pos1, ubo.h) * (pressure2 / (density2 * density2) + pressure1 / (density1 * density1)) * 0.5 * mass2;

                if pos1.y < ubo.h - boxR {
                    pos3.y = -2.0 * boxR - pos2.y;
                    F_pressure += W_spiky_grad(pos3 - pos1, ubo.h) * (pressure2 / (density2 * density2) + pressure1 / (density1 * density1)) * 0.5 * mass2;
                }
            } else if pos1.x > boxR - ubo.h {
                pos3.x = 2.0 * boxR - pos2.x;
                F_pressure += W_spiky_grad(pos3 - pos1, ubo.h) * (pressure2 / (density2 * density2) + pressure1 / (density1 * density1)) * 0.5 * mass2;

                if pos1.y < ubo.h - boxR {
                    pos3.y = -2.0 * boxR - pos2.y;
                    F_pressure += W_spiky_grad(pos3 - pos1, ubo.h) * (pressure2 / (density2 * density2) + pressure1 / (density1 * density1)) * 0.5 * mass2;
                }
            } else if pos1.y < ubo.h - boxR {
                pos3.y = -2.0 * boxR - pos2.y;
                F_pressure += W_spiky_grad(pos3 - pos1, ubo.h) * (pressure2 / (density2 * density2) + pressure1 / (density1 * density1)) * 0.5 * mass2;
            }
        }
    }

    var F_net = vec2<f32>(0.0, -8.0);
    F_net += F_pressure;
    F_net += 0.4 * F_viscosity;
    F_net += 2.0 * F_cohestion;
    F_net += 1.3 * F_curvature;

    acc[gid] = F_net / density1;
}
