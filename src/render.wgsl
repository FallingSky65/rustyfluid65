struct Uniform {
    canvas_size: vec2<f32>,
    res: vec2<f32>,
};
@group(0) @binding(0) var<uniform> ubo: Uniform;

struct vertex_in {
    @location(0) position: vec2<f32>,
    // @builtin(vertex_index) vertex_index: u32,
};

struct instance_in {
    @location(1) position: vec4<f32>,
    @location(2) color: vec4<f32>,
};

struct vs_out {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec3<f32>,
};

const RADIUS: f32 = 07.0;

@vertex
fn vs_main(
    v_in: vertex_in,
    i_in: instance_in,
) -> vs_out {
    var out: vs_out;
    out.position = i_in.position.xy;
    out.uv = v_in.position;
    out.color = i_in.color.rgb;

    var pos: vec2<f32> = i_in.position.xy + v_in.position * RADIUS;
    pos = (pos * 2.0 - ubo.canvas_size) / ubo.canvas_size;

    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: vs_out) -> @location(0) vec4<f32> {
    if length(in.uv) > 1.0 {
        discard;
    }
    return vec4<f32>(in.color, 1.0);
}
