@group(0) @binding(0)
var canvas: texture_2d<f32>;
@group(0) @binding(1)
var canvas_sampler: sampler;

struct vertex_in {
    @location(0) position: vec2<f32>,
    // @builtin(vertex_index) vertex_index: u32,
};

struct vs_out {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

const RADIUS: f32 = 10.0;

@vertex
fn vs_main(
    v_in: vertex_in,
) -> vs_out {
    var out: vs_out;
    out.uv = v_in.position * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    out.clip_position = vec4<f32>(v_in.position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: vs_out) -> @location(0) vec4<f32> {
    return textureSample(canvas, canvas_sampler, in.uv);
}
