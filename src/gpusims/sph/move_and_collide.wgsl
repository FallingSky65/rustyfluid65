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
@group(0) @binding(1) var<storage, read_write> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> acc: array<vec2<f32>>;
struct instance {
    position: vec4<f32>,
    color: vec4<f32>,
};
@group(0) @binding(4) var<storage, read_write> instances: array<instance>;

//@group(0) @binding(5) var<storage, read> hashes1: array<vec2<u32>>;
//@group(0) @binding(6) var<storage, read> cell_ranges: array<vec2<u32>>;

override WORKGROUP_SIZE: u32 = 256u;

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    const K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

fn gradient(t: f32) -> vec3<f32> {
    var t1 = t / 7.0;
    t1 = clamp(1.0 - t1, 0.0, 1.0);
    return hsv2rgb(vec3<f32>(t1 * 2.0 / 3.0, 1.0, 1.0));
}

@compute
@workgroup_size(WORKGROUP_SIZE)
fn move_and_collide(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let gid = global_id.x;
    if gid >= ubo.n {
        return;
    }

    //pos[gid] -= vel[gid] * ubo.dt;
    vel[gid] += acc[gid] * ubo.dt;
    pos[gid] += vel[gid] * ubo.dt;

    var p = pos[gid];
    var v = vel[gid];
    //var a = acc[gid];
    
    const bounce: f32 = 0.5;
    const box_r: f32 = 9.9;
    v.x = select(
        v.x,
        v.x * -bounce,
        p.x < -box_r || p.x > box_r
    );
    v.y = select(
        v.y,
        v.y * -bounce,
        p.y < -box_r || p.y > box_r
    );
    p.x = select(
        select(
            p.x,
            box_r,
            p.x > box_r,
        ),
        -box_r,
        p.x < -box_r
    );
    p.y = select(
        select(
            p.y,
            box_r,
            p.y > box_r,
        ),
        -box_r,
        p.y < -box_r
    );

    pos[gid] = p;
    vel[gid] = v;

    let halfdims = vec2(ubo.canvas_width, ubo.canvas_height) * 0.5;
    instances[gid].position = vec4((p / 10.0) * halfdims + halfdims, 0, 0);
    instances[gid].color = vec4(gradient(length(v)), 1);
    //instances[gid].color = select(vec4(1.0, 0.0, 0.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), cell_ranges[gid].x == 0);
    //instances[gid].color = select(vec4(1.0, 0.0, 0.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), gid == 0 || hashes1[gid].x >= hashes1[gid - 1].x);
}
