struct UBO {
    NPARTICLES: u32,
    smooth_radius: f32,
    block_size: u32,
    flip: u32,
    gas_constant: f32,
    target_density: f32,
    canvas_width: f32,
    canvas_height: f32,
    dt: f32,
};
@group(0) @binding(0) var<uniform> ubo: UBO;

struct particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    acc: vec2<f32>,
    normal: vec2<f32>,
    mass: f32,
    density: f32,
    pressure: f32,
    _pad: f32,
}
@group(1) @binding(0) var<storage, read_write> particles: array<particle>;

@group(1) @binding(1) var<storage, read_write> particle_hash: array<vec2<u32>>;
@group(1) @binding(2) var<storage, read_write> cell_range: array<vec2<u32>>;

struct instance {
    position: vec4<f32>,
    color: vec4<f32>,
};
@group(1) @binding(3) var<storage, read_write> instances: array<instance>;

const neighbors: array<vec2<i32>, 9> = array(
    vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1),
    vec2<i32>(-1, 0), vec2<i32>(0, 0), vec2<i32>(1, 0),
    vec2<i32>(-1, 1), vec2<i32>(0, 1), vec2<i32>(1, 1),
);

fn hash2D(v: vec2<i32>) -> u32 {
    return bitcast<u32>(v.x * 18397 + v.y * 20483);
}

@compute
@workgroup_size(256)
fn calc_hash(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if gid.x >= ubo.NPARTICLES {
        return;
    }

    particle_hash[gid.x] = vec2<u32>(
        hash2D(vec2<i32>(floor(particles[gid.x].pos / ubo.smooth_radius))) % ubo.NPARTICLES, gid.x
    );
}

@compute
@workgroup_size(256)
fn bitonic_sort(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let block_size = ubo.block_size;
    let block_index = gid.x / (block_size / 2);
    let pos_in_block = gid.x % (block_size / 2);

    let a = block_index * block_size + pos_in_block;
    if a >= ubo.NPARTICLES {
        return;
    }
    let b = select(a + block_size / 2, (block_index + 1) * block_size - pos_in_block - 1, ubo.flip > 0);
    if b >= ubo.NPARTICLES {
        return;
    }

    if particle_hash[a].x > particle_hash[b].x {
        let temp = particle_hash[a];
        particle_hash[a] = particle_hash[b];
        particle_hash[b] = temp;
    }
}

@compute
@workgroup_size(256)
fn find_range(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if gid.x >= ubo.NPARTICLES {
        return;
    }

    if gid.x == 0 || particle_hash[gid.x].x != particle_hash[gid.x - 1].x {
        cell_range[particle_hash[gid.x].x].x = gid.x;
    }
    if gid.x == ubo.NPARTICLES - 1 || particle_hash[gid.x].x != particle_hash[gid.x + 1].x {
        cell_range[particle_hash[gid.x].x].y = gid.x + 1;
    }
}

const pi: f32 = 3.141592653589793238462643;

fn W_poly6(r: f32, h: f32) -> f32 {
    const factor2d = 4.0 / pi;
    return factor2d / pow(h, 8) * select(
        0.0,
        pow(h * h - r * r, 3),
        r <= h
    );
}

fn sample_density(v: vec2<f32>) -> f32 {
    var density: f32 = 1.0;
    let cell_pos = vec2<i32>(floor(v / ubo.smooth_radius));

    for (var i: u32 = 0; i < 9u; i += 1) {
        let cell: vec2<i32> = cell_pos + neighbors[i];
        let range: vec2<u32> = cell_range[hash2D(cell) % ubo.NPARTICLES];
        for (var j: u32 = 0; j < range.y; j += 1) {
            let particle = particles[particle_hash[j].y];
            density += particle.mass * W_poly6(length(particle.pos - v), ubo.smooth_radius);
            continue;

            var ghost = particle;
            ghost.pos.x = -20 - particle.pos.x;
            density += ghost.mass * W_poly6(length(ghost.pos - v), ubo.smooth_radius);
            ghost.pos.x = 20 - particle.pos.x;
            density += ghost.mass * W_poly6(length(ghost.pos - v), ubo.smooth_radius);
            ghost.pos.x = particle.pos.x;
            ghost.pos.y = -20 - particle.pos.y;
            density += ghost.mass * W_poly6(length(ghost.pos - v), ubo.smooth_radius);
        }
    }

    return density;
}

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

fn sample_pressure_force(v: vec2<f32>, pdensity: f32, ppressure: f32) -> vec2<f32> {
    var pressure_force = vec2<f32>(0.0);
    let cell_pos = vec2<i32>(floor(v / ubo.smooth_radius));

    for (var i: u32 = 0; i < 9u; i += 1) {
        let cell: vec2<i32> = cell_pos + neighbors[i];
        let range: vec2<u32> = cell_range[hash2D(cell) % ubo.NPARTICLES];
        for (var j: u32 = 0; j < range.y; j += 1) {
            let particle = particles[particle_hash[j].y];
            pressure_force += W_spiky_grad(particle.pos - v, ubo.smooth_radius) * (particle.pressure / pow(particle.density, 2) + ppressure / pow(pdensity, 2)) * 0.5 * particle.mass;
            continue;

            var ghost = particle;
            ghost.pos.x = -20 - particle.pos.x;
            pressure_force += W_spiky_grad(ghost.pos - v, ubo.smooth_radius) * (ghost.pressure + ppressure) * 0.5 * ghost.mass / ghost.density;
            ghost.pos.x = 20 - particle.pos.x;
            pressure_force += W_spiky_grad(ghost.pos - v, ubo.smooth_radius) * (ghost.pressure + ppressure) * 0.5 * ghost.mass / ghost.density;
            ghost.pos.x = particle.pos.x;
            ghost.pos.y = -20 - particle.pos.y;
            pressure_force += W_spiky_grad(ghost.pos - v, ubo.smooth_radius) * (ghost.pressure + ppressure) * 0.5 * ghost.mass / ghost.density;
        }
    }

    return pressure_force;
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

fn sample_viscosity_force(v: vec2<f32>, vel: vec2<f32>) -> vec2<f32> {
    var viscosity_force = vec2<f32>(0.0);
    let cell_pos = vec2<i32>(floor(v / ubo.smooth_radius));

    for (var i: u32 = 0; i < 9u; i += 1) {
        let cell: vec2<i32> = cell_pos + neighbors[i];
        let range: vec2<u32> = cell_range[hash2D(cell) % ubo.NPARTICLES];
        for (var j: u32 = 0; j < range.y; j += 1) {
            let particle = particles[particle_hash[j].y];
            var dst = length(v - particle.pos);
            if dst != 0 {
                viscosity_force += (particle.mass / particle.density) * 2 * (vel - particle.vel) / dst * W_cubic_deriv(dst, ubo.smooth_radius);
                continue;

                var ghost = particle;
                ghost.pos.x = -20 - particle.pos.x;
                ghost.vel.x = -particle.vel.x;
                dst = length(v - ghost.pos);
                if dst != 0 {
                    viscosity_force += (ghost.mass / ghost.density) * 2 * (vel - ghost.vel) / dst * W_cubic_deriv(dst, ubo.smooth_radius);
                }
                ghost.pos.x = 20 - particle.pos.x;
                dst = length(v - ghost.pos);
                if dst != 0 {
                    viscosity_force += (ghost.mass / ghost.density) * 2 * (vel - ghost.vel) / dst * W_cubic_deriv(dst, ubo.smooth_radius);
                }
                ghost.pos.x = particle.pos.x;
                ghost.pos.y = -20 - particle.pos.y;
                ghost.vel.x = particle.vel.x;
                ghost.vel.y = -particle.vel.y;
                dst = length(v - ghost.pos);
                if dst != 0 {
                    viscosity_force += (ghost.mass / ghost.density) * 2 * (vel - ghost.vel) / dst * W_cubic_deriv(dst, ubo.smooth_radius);
                }
            }
        }
    }

    return viscosity_force;
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

fn sample_cohesion_force(v: vec2<f32>, mass: f32) -> vec2<f32> {
    var cohesion_force = vec2<f32>(0.0);
    let cell_pos = vec2<i32>(floor(v / ubo.smooth_radius));

    for (var i: u32 = 0; i < 9u; i += 1) {
        let cell: vec2<i32> = cell_pos + neighbors[i];
        let range: vec2<u32> = cell_range[hash2D(cell) % ubo.NPARTICLES];
        for (var j: u32 = 0; j < range.y; j += 1) {
            let particle = particles[particle_hash[j].y];
            if length(v - particle.pos) > 0 {
                cohesion_force -= mass * W_cohesion(length(v - particle.pos), ubo.smooth_radius) * normalize(v - particle.pos);
                continue;

                var ghost = particle;
                ghost.pos.x = -20 - particle.pos.x;
                cohesion_force -= mass * W_cohesion(length(v - ghost.pos), ubo.smooth_radius) * normalize(v - ghost.pos);
                ghost.pos.x = 20 - particle.pos.x;
                cohesion_force -= mass * W_cohesion(length(v - ghost.pos), ubo.smooth_radius) * normalize(v - ghost.pos);
                ghost.pos.x = particle.pos.x;
                ghost.pos.y = -20 - particle.pos.y;
                cohesion_force -= mass * W_cohesion(length(v - ghost.pos), ubo.smooth_radius) * normalize(v - ghost.pos);
            }
        }
    }

    return cohesion_force;
}

fn sample_curvature_force(v: vec2<f32>, n: vec2<f32>, mass: f32) -> vec2<f32> {
    var curvature_force = vec2<f32>(0.0);
    let cell_pos = vec2<i32>(floor(v / ubo.smooth_radius));

    for (var i: u32 = 0; i < 9u; i += 1) {
        let cell: vec2<i32> = cell_pos + neighbors[i];
        let range: vec2<u32> = cell_range[hash2D(cell) % ubo.NPARTICLES];
        for (var j: u32 = 0; j < range.y; j += 1) {
            let particle = particles[particle_hash[j].y];
            if length(v - particle.pos) < ubo.smooth_radius {
                curvature_force += (n - particle.normal);
            }
        }
    }

    return curvature_force * mass;
}

@compute
@workgroup_size(256)
fn set_position(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if gid.x >= ubo.NPARTICLES {
        return;
    }

    particles[gid.x].pos += particles[gid.x].vel * ubo.dt;
}

@compute
@workgroup_size(256)
fn set_density_pressure(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if gid.x >= ubo.NPARTICLES {
        return;
    }
    particles[gid.x].density = sample_density(particles[gid.x].pos);
    //particles[gid.x].pressure = ubo.gas_constant * (particles[gid.x].density - ubo.target_density);
    particles[gid.x].pressure = max(ubo.gas_constant * (pow(particles[gid.x].density / ubo.target_density, 7) - 1), 0);
    // particles[gid.x].pressure = ubo.gas_constant * (pow(particles[gid.x].density / ubo.target_density, 7) - 1);
}

@compute
@workgroup_size(256)
fn set_normal(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if gid.x >= ubo.NPARTICLES {
        return;
    }

    let v = particles[gid.x].pos;
    var normal = vec2<f32>(0.0);
    let cell_pos = vec2<i32>(floor(v / ubo.smooth_radius));

    for (var i: u32 = 0; i < 9u; i += 1) {
        let cell: vec2<i32> = cell_pos + neighbors[i];
        let range: vec2<u32> = cell_range[hash2D(cell) % ubo.NPARTICLES];
        for (var j: u32 = 0; j < range.y; j += 1) {
            let particle = particles[particle_hash[j].y];
            if length(v - particle.pos) > 0 {
                normal += ubo.smooth_radius * particle.mass / particle.density * W_cubic_deriv(length(v - particle.pos), ubo.smooth_radius) * normalize(particle.pos - v);
                continue;

                var ghost = particle;
                ghost.pos.x = -20 - particle.pos.x;
                normal += ubo.smooth_radius * ghost.mass / ghost.density * W_cubic_deriv(length(v - ghost.pos), ubo.smooth_radius) * normalize(ghost.pos - v);
                ghost.pos.x = 20 - particle.pos.x;
                normal += ubo.smooth_radius * ghost.mass / ghost.density * W_cubic_deriv(length(v - ghost.pos), ubo.smooth_radius) * normalize(ghost.pos - v);
                ghost.pos.x = particle.pos.x;
                ghost.pos.y = -20 - particle.pos.y;
                normal += ubo.smooth_radius * ghost.mass / ghost.density * W_cubic_deriv(length(v - ghost.pos), ubo.smooth_radius) * normalize(ghost.pos - v);
            }
        }
    }

    particles[gid.x].normal = normal;
}

@compute
@workgroup_size(256)
fn set_acceleration(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if gid.x >= ubo.NPARTICLES {
        return;
    }

    var F_net = vec2<f32>(0.0, -8.0);
    // var F_net = -particles[gid.x].pos;
    F_net += sample_pressure_force(particles[gid.x].pos, particles[gid.x].density, particles[gid.x].pressure);
    F_net += 3.0 * sample_viscosity_force(particles[gid.x].pos, particles[gid.x].vel);
    F_net += 2.0 * sample_cohesion_force(particles[gid.x].pos, particles[gid.x].mass);
    F_net -= 0.3 * sample_curvature_force(particles[gid.x].pos, particles[gid.x].normal, particles[gid.x].mass);

    particles[gid.x].acc = F_net / particles[gid.x].density;
}

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
@workgroup_size(256)
fn move_and_collide(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if gid.x >= ubo.NPARTICLES {
        return;
    }
    // particles[gid.x].pos -= particles[gid.x].vel * ubo.dt;
    // particles[gid.x].vel += particles[gid.x].acc * ubo.dt;
    // particles[gid.x].pos += particles[gid.x].vel * ubo.dt;

    var pos = particles[gid.x].pos;
    var vel = particles[gid.x].vel;

    const damping: f32 = 0.2;
    const r: f32 = 10.0;
    if pos.x < -r {
        pos.x = -r;
        if vel.x < 0.0 {
            vel.x *= -damping;
        }
    }
    if pos.y < -r {
        pos.y = -r;
        if vel.y < 0.0 {
            vel.y *= -damping;
        }
    }
    if pos.x > r {
        pos.x = r;
        if vel.x > 0.0 {
            vel.x *= -damping;
        }
    }
    if pos.y > r {
        pos.y = r;
        if vel.y > 0.0 {
            vel.y *= -damping;
        }
    }

    particles[gid.x].pos = pos;
    particles[gid.x].vel = vel;

    let halfdims = vec2(ubo.canvas_width, ubo.canvas_height) * 0.5;
    instances[gid.x].position = vec4((particles[gid.x].pos / 10.0) * halfdims + halfdims, 0, 0);
    instances[gid.x].color = vec4(gradient(length(particles[gid.x].vel)), 1);
}
