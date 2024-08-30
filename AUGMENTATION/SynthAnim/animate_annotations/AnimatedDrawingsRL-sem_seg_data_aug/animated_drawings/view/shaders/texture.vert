// Copyright (c) Meta Platforms, Inc. and affiliates.

#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 3) in vec2 texCoord;

out vec3 ourColor;
out vec2 TexCoord;
out vec4 Position;

uniform mat4 model;
uniform mat4 proj_view;

uniform bool reflection_flag;

void main() {
    vec4 _pos = model * vec4(pos, 1.0);
    if(reflection_flag){
        _pos[1] = -_pos[1];

        // reflections shouldn't come up out of floor
        if(_pos[1] > 0.0){
            _pos[1] = 0.0;
        }
    };

    gl_Position = proj_view * _pos;

    TexCoord = texCoord;

    Position = _pos;
}
