// Copyright (c) Meta Platforms, Inc. and affiliates.

#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec4 Position;

uniform sampler2D texture_sampler;

void main() {
    vec4 color = texture(texture_sampler, TexCoord);

    if (color.a < 0.1){
        discard;
    }

    // make character reflections a little see through
    if (Position.y < 0.0){
        color.a = 0.7;
    }

    FragColor = color;
}
