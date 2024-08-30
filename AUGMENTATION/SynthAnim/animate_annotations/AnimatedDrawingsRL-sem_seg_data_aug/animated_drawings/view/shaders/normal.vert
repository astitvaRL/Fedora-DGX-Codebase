// Copyright (c) Meta Platforms, Inc. and affiliates.

#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 2) in vec3 aNormal;

out vec3 ourColor;

uniform mat4 model;
uniform mat4 proj_view;

void main() {
    gl_Position = proj_view * model * vec4(pos, 1.0);
    ourColor = vec3(model * vec4(aNormal, 0.0));
}
