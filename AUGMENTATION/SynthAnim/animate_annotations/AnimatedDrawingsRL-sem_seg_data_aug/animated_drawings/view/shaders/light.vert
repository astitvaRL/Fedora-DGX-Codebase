#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aNormal;
layout(location = 3) in vec2 texCoord;

uniform mat4 model;
uniform mat4 proj_view;
uniform mat4 shadow_view_proj_matrix;

out VS_OUT {
    vec3 frag_pos;
    vec3 normal;
    vec3 color;
    vec2 tex_coords;
    vec4 shadow_coord;
} vs_out;

void main() {
    vs_out.frag_pos = vec3(model * vec4(aPos, 1.0));
    vs_out.normal = vec3(model * vec4(aNormal, 0.0));
    vs_out.color = aColor;

    vs_out.shadow_coord = shadow_view_proj_matrix * model * vec4(aPos, 1.0);

    vs_out.tex_coords = texCoord;

    gl_Position = proj_view * model * vec4(aPos, 1.0);
}