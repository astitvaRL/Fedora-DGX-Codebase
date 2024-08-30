#version 330 core

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

in VS_OUT {
    vec3 frag_pos;
    vec3 normal;
    vec3 color;
    vec2 tex_coords;
    vec4 shadow_coord;
} fs_in;

uniform sampler2D depth_texture;  // shadow map

uniform vec3 viewPos;
uniform Material material;
uniform Light light;

out vec4 FragColor;

float ShadowCalculation(vec4 shadow_coord){
    //perspective divide
    vec3 proj_coords = shadow_coord.xyz / shadow_coord.w;
    // transform to [0,1] range
    proj_coords = proj_coords * 0.5 + 0.5;
    // get closest depth value from light's perspective 
    float closest_depth = texture(depth_texture, proj_coords.xy).r;

    // check whether current frag pos is in shadow
    float current_depth = proj_coords.z;
    float bias = 0.0000002;
    float shadow = closest_depth < (current_depth- bias) ? 1.0 : 0.0;

    return shadow;
}

void main() {
    // ambient
    vec3 ambient = light.ambient * fs_in.color;

    // diffuse
    vec3 norm = normalize(fs_in.normal);
    vec3 lightDir = normalize(light.position - fs_in.frag_pos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * (diff * fs_in.color);

    // specular
    vec3 viewDir = normalize(viewPos - fs_in.frag_pos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);  // everything is the same amount of shiny for now
    vec3 specular =  light.specular * (spec * fs_in.color);

    float shadow = ShadowCalculation(fs_in.shadow_coord);
    vec3 result = ambient + (1.0 - shadow) * (diffuse + specular);
    FragColor = vec4(result, 1.0);
}