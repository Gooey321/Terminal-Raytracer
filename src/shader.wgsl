struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
    _padding: f32,
};

struct Primitive {
    primitive_type: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
    
    color: Vec3,
    emission: Vec3,
    reflectivity: f32,
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
    
    sphere_center: Vec3,
    sphere_radius: f32,
    _sphere_padding: array<f32, 3>,
    
    plane_point: Vec3,
    plane_normal: Vec3,
    
    triangle_v0: Vec3,
    triangle_v1: Vec3,
    triangle_v2: Vec3,
    _triangle_padding: f32,
};

struct Ray {
    origin: Vec3,
    direction: Vec3,
};

struct HitRecord {
    p: Vec3,
    normal: Vec3,
    t: f32,
    front_face: bool,
    color: Vec3,
    emission: Vec3,
    reflectivity: f32,
};

struct SunConfig {
    direction: Vec3,
    color: Vec3,
    intensity: f32,
    angular_size: f32,
    _padding1: f32,
    _padding2: f32,
};

struct Uniforms {
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    seed: u32,
    frame_number: u32,
    _padding1: u32,
    _padding2: u32,
    aspect_ratio: f32,
    char_aspect_ratio: f32,
    fov_rad: f32,
    _padding3: f32,
    camera_pos: Vec3,
    camera_forward: Vec3,
    camera_right: Vec3,
    camera_up: Vec3,
};

struct VarianceBuffer {
    variance: f32,
    sample_count: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> primitives: array<Primitive>;
@group(0) @binding(2) var<storage, read_write> pixels: array<Vec3>;
@group(0) @binding(3) var<storage, read_write> accumulation: array<Vec3>;
@group(0) @binding(4) var<storage, read_write> variance: array<VarianceBuffer>;

// Vector operations
fn vec3_add(a: Vec3, b: Vec3) -> Vec3 {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z, 0.0);
}

fn vec3_sub(a: Vec3, b: Vec3) -> Vec3 {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z, 0.0);
}

fn vec3_mul(a: Vec3, s: f32) -> Vec3 {
    return Vec3(a.x * s, a.y * s, a.z * s, 0.0);
}

fn vec3_mul_vec3(a: Vec3, b: Vec3) -> Vec3 {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z, 0.0);
}

fn vec3_div(a: Vec3, s: f32) -> Vec3 {
    return Vec3(a.x / s, a.y / s, a.z / s, 0.0);
}

// Utility functions
var<private> rand_state: u32;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn random_f32() -> f32 {
    rand_state = pcg_hash(rand_state);
    return f32(rand_state) / 4294967295.0;
}

fn random_in_unit_sphere() -> Vec3 {
    for (var i = 0; i < 100; i = i + 1) {
        let p = Vec3(random_f32() * 2.0 - 1.0, random_f32() * 2.0 - 1.0, random_f32() * 2.0 - 1.0, 0.0);
        if (dot(p,p) < 1.0) { return p; }
    }
    // Fallback if we don't find a point in the sphere after 100 tries
    return Vec3(0.0, 1.0, 0.0, 0.0);
}

fn dot(a: Vec3, b: Vec3) -> f32 { 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

fn length(v: Vec3) -> f32 { 
    return sqrt(dot(v, v)); 
}

fn normalize(v: Vec3) -> Vec3 { 
    return vec3_div(v, length(v)); 
}

fn reflect(v: Vec3, n: Vec3) -> Vec3 { 
    return vec3_sub(v, vec3_mul(n, 2.0 * dot(v, n))); 
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
        0.0
    );
}

fn clamp(value: f32, min_val: f32, max_val: f32) -> f32 {
    return max(min_val, min(max_val, value));
}

fn acos(x: f32) -> f32 {
    return atan2(sqrt(1.0 - x * x), x);
}

fn sample_light_direction(hit_point: Vec3, primitive: Primitive) -> Vec3 {
    if (primitive.primitive_type == 0u) {
        let to_light = vec3_sub(primitive.sphere_center, hit_point);
        let distance = length(to_light);
        let light_dir = vec3_div(to_light, distance);
        let random_offset = vec3_mul(random_in_unit_sphere(), primitive.sphere_radius);
        return normalize(vec3_add(light_dir, random_offset));
    }
    return Vec3(0.0, 1.0, 0.0, 0.0); // FIX: Added the 4th component for padding
}

// Ray-sphere intersection
fn hit_sphere(ray: Ray, primitive: Primitive, t_min: f32, t_max: f32) -> f32 {
    let oc = vec3_sub(primitive.sphere_center, ray.origin);
    let a = dot(ray.direction, ray.direction);
    let h = dot(ray.direction, oc);
    let c = dot(oc, oc) - primitive.sphere_radius * primitive.sphere_radius;
    let discriminant = h * h - a * c;

    if (discriminant < 0.0) {
        return -1.0;
    }

    let sqrtd = sqrt(discriminant);
    var root = (h - sqrtd) / a;
    if (root <= t_min || t_max <= root) {
        root = (h + sqrtd) / a;
        if (root <= t_min || t_max <= root) {
            return -1.0;
        }
    }
    return root;
}

// Ray-plane intersection
fn hit_plane(ray: Ray, primitive: Primitive, t_min: f32, t_max: f32) -> f32 {
    let denom = dot(primitive.plane_normal, ray.direction);
    if (abs(denom) < 0.0001) {
        return -1.0; // Ray is parallel to plane
    }
    
    let t = dot(vec3_sub(primitive.plane_point, ray.origin), primitive.plane_normal) / denom;
    if (t < t_min || t > t_max) {
        return -1.0;
    }
    return t;
}

// Ray-triangle intersection using Möller-Trumbore algorithm
fn hit_triangle(ray: Ray, primitive: Primitive, t_min: f32, t_max: f32) -> f32 {
    let edge1 = vec3_sub(primitive.triangle_v1, primitive.triangle_v0);
    let edge2 = vec3_sub(primitive.triangle_v2, primitive.triangle_v0);
    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);
    
    if (a > -0.00001 && a < 0.00001) {
        return -1.0; // Ray is parallel to triangle
    }
    
    let f = 1.0 / a;
    let s = vec3_sub(ray.origin, primitive.triangle_v0);
    let u = f * dot(s, h);
    
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }
    
    let q = cross(s, edge1);
    let v = f * dot(ray.direction, q);
    
    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }
    
    let t = f * dot(edge2, q);
    if (t > t_min && t < t_max) {
        return t;
    }
    
    return -1.0;
}

fn get_normal_at_hit(ray: Ray, primitive: Primitive, hit_point: Vec3) -> Vec3 {
    if (primitive.primitive_type == 0u) { // Sphere
        return normalize(vec3_div(vec3_sub(hit_point, primitive.sphere_center), primitive.sphere_radius));
    } else if (primitive.primitive_type == 1u) { // Plane
        return normalize(primitive.plane_normal);
    } else if (primitive.primitive_type == 2u) { // Triangle
        let edge1 = vec3_sub(primitive.triangle_v1, primitive.triangle_v0);
        let edge2 = vec3_sub(primitive.triangle_v2, primitive.triangle_v0);
        return normalize(cross(edge1, edge2));
    }
    return Vec3(0.0, 1.0, 0.0, 0.0); // Default normal
}

// Updated hit_scene function
fn hit_scene(ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
    var closest_so_far = t_max;
    var hit: HitRecord;
    hit.t = -1.0;

    for (var i = 0u; i < arrayLength(&primitives); i = i + 1) {
        let primitive = primitives[i];
        var t = -1.0;
        
        if (primitive.primitive_type == 0u) { // Sphere
            t = hit_sphere(ray, primitive, t_min, closest_so_far);
        } else if (primitive.primitive_type == 1u) { // Plane
            t = hit_plane(ray, primitive, t_min, closest_so_far);
        } else if (primitive.primitive_type == 2u) { // Triangle
            t = hit_triangle(ray, primitive, t_min, closest_so_far);
        }
        
        if (t > 0.0 && t < closest_so_far) {
            closest_so_far = t;
            hit.t = t;
            hit.p = vec3_add(ray.origin, vec3_mul(ray.direction, t));
            hit.normal = get_normal_at_hit(ray, primitive, hit.p);
            hit.front_face = dot(ray.direction, hit.normal) < 0.0;
            if (!hit.front_face) {
                hit.normal = vec3_mul(hit.normal, -1.0);
            }
            hit.color = primitive.color;
            hit.emission = primitive.emission;
            hit.reflectivity = primitive.reflectivity;
        }
    }
    return hit;
}

// Replace the get_sky_color function with a simple background
fn get_background_color(direction: Vec3) -> Vec3 {
    // Simple solid color background (dark gray/black)
    return Vec3(0.0, 0.0, 0.0, 0.0);
}

// Update the ray_color function
fn ray_color(initial_ray: Ray) -> Vec3 {
    var accumulated_color = Vec3(0.0, 0.0, 0.0, 0.0);
    var attenuation = Vec3(1.0, 1.0, 1.0, 0.0);
    var current_ray = initial_ray;

    for (var i = 0u; i < uniforms.max_depth; i = i + 1) {
        let hit = hit_scene(current_ray, 0.001, 1e10);

        if (hit.t < 0.0) {
            return vec3_add(accumulated_color, vec3_mul_vec3(get_background_color(current_ray.direction), attenuation));
        }

        // Add emission from directly hit emissive surfaces
        accumulated_color = vec3_add(accumulated_color, vec3_mul_vec3(hit.emission, attenuation));
        
        // NEXT EVENT ESTIMATION: Direct lighting calculation
        // Sample all light sources for direct illumination
        for (var light_idx = 0u; light_idx < arrayLength(&primitives); light_idx = light_idx + 1) {
            let light_primitive = primitives[light_idx];
            
            // Check if this primitive is a light source
            let light_power = light_primitive.emission.x + light_primitive.emission.y + light_primitive.emission.z;
            if (light_power > 0.001) {
                
                // Sample a point on the light source
                var light_point: Vec3;
                var light_normal: Vec3;
                var light_area: f32;
                
                if (light_primitive.primitive_type == 0u) { // Sphere light
                    // Sample random point on sphere surface
                    let random_dir = normalize(random_in_unit_sphere());
                    light_point = vec3_add(light_primitive.sphere_center, 
                                         vec3_mul(random_dir, light_primitive.sphere_radius));
                    light_normal = random_dir;
                    light_area = 4.0 * 3.14159 * light_primitive.sphere_radius * light_primitive.sphere_radius;
                } else {
                    continue; // Skip non-sphere lights for simplicity
                }
                
                // Calculate light direction and distance
                let light_vec = vec3_sub(light_point, hit.p);
                let light_distance = length(light_vec);
                let light_dir = vec3_div(light_vec, light_distance);
                
                // Check visibility with shadow ray
                let shadow_ray = Ray(hit.p, light_dir);
                let shadow_hit = hit_scene(shadow_ray, 0.001, light_distance - 0.001);
                
                // If not in shadow, calculate direct lighting contribution
                if (shadow_hit.t < 0.0) {
                    let cos_theta_surface = max(0.0, dot(hit.normal, light_dir));
                    let cos_theta_light = max(0.0, dot(light_normal, vec3_mul(light_dir, -1.0)));
                    
                    if (cos_theta_surface > 0.0 && cos_theta_light > 0.0) {
                        // Calculate geometric term (inverse square law + cosine terms)
                        let distance_sq = light_distance * light_distance;
                        let geometric_term = (cos_theta_surface * cos_theta_light) / distance_sq;
                        
                        // BRDF evaluation (Lambertian diffuse)
                        let brdf = vec3_div(hit.color, 3.14159); // Lambertian BRDF = albedo/π
                        
                        // Calculate direct lighting contribution
                        let direct_light = vec3_mul_vec3(
                            vec3_mul_vec3(brdf, light_primitive.emission),
                            vec3_mul(attenuation, geometric_term * light_area)
                        );
                        
                        accumulated_color = vec3_add(accumulated_color, direct_light);
                    }
                }
            }
        }
        
        // INDIRECT LIGHTING: Sample next bounce direction
        var scatter_direction: Vec3;
        var pdf: f32;
        
        let is_reflective = hit.reflectivity > random_f32();
        
        if (is_reflective) {
            // Perfect specular reflection
            scatter_direction = reflect(current_ray.direction, hit.normal);
            pdf = 1.0; // Delta function
        } else {
            // Cosine-weighted hemisphere sampling (importance sampling for Lambertian BRDF)
            let random1 = random_f32();
            let random2 = random_f32();
            
            // Generate cosine-weighted direction
            let cos_theta = sqrt(random1);
            let sin_theta = sqrt(1.0 - random1);
            let phi = 2.0 * 3.14159 * random2;
            
            // Local coordinates (normal = z-axis)
            let local_dir = Vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta, 0.0);
            
            var up: Vec3;
            if (abs(hit.normal.x) > 0.9) {
                up = Vec3(1.0, 0.0, 0.0, 0.0);
            } else {
                up = Vec3(0.0, 1.0, 0.0, 0.0);
            }
            let tangent = normalize(cross(up, hit.normal));
            let bitangent = cross(hit.normal, tangent);
            
            scatter_direction = normalize(vec3_add(
                vec3_add(
                    vec3_mul(tangent, local_dir.x),
                    vec3_mul(bitangent, local_dir.y)
                ),
                vec3_mul(hit.normal, local_dir.z)
            ));
            
            pdf = cos_theta / 3.14159; // PDF for cosine-weighted sampling
        }
        
        // Update attenuation for next bounce
        if (is_reflective) {
            attenuation = vec3_mul_vec3(attenuation, hit.color);
        } else {
            // For Lambertian surfaces: BRDF * cos(theta) / PDF
            let cos_theta = max(0.0, dot(hit.normal, scatter_direction));
            let brdf_value = vec3_div(hit.color, 3.14159);
            attenuation = vec3_mul_vec3(attenuation, vec3_mul(brdf_value, cos_theta / pdf));
        }
        
        current_ray = Ray(hit.p, scatter_direction);

        // Russian Roulette path termination (more aggressive for realism)
        if (i > 1u) {
            let survival_prob = min(0.95, max(attenuation.x, max(attenuation.y, attenuation.z)));
            if (survival_prob < random_f32()) {
                break;
            }
            attenuation = vec3_div(attenuation, survival_prob);
        }
    }
    return accumulated_color;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }
    
    rand_state = (y * uniforms.width + x) * 1973u + uniforms.seed * 9277u + uniforms.frame_number * 12345u;
    
    var pixel_color = Vec3(0.0, 0.0, 0.0, 0.0);
    var color_sum = Vec3(0.0, 0.0, 0.0, 0.0);
    var color_squared_sum = Vec3(0.0, 0.0, 0.0, 0.0);

    let base_samples = max(4u, uniforms.samples_per_pixel / 4u);
    let max_samples = uniforms.samples_per_pixel;

    for (var i = 0u; i < base_samples; i = i + 1) {
        rand_state = pcg_hash(rand_state + i * 5096u);

        let u = (f32(x) + random_f32()) / f32(uniforms.width - 1u);
        let v = (f32(uniforms.height - 1u - y) + random_f32()) / f32(uniforms.height - 1u);

        let half_height = tan(f32(uniforms.fov_rad) / 2.0);
        let half_width = f32(uniforms.aspect_ratio) * half_height;
        let viewport_x = half_width * (2.0 * u - 1.0);
        let viewport_y = half_height * (2.0 * v - 1.0) / f32(uniforms.char_aspect_ratio);

        let direction = normalize(
            vec3_add(
                vec3_add(
                    uniforms.camera_forward,
                    vec3_mul(uniforms.camera_right, viewport_x)
                ),
                vec3_mul(uniforms.camera_up, viewport_y)
            )
        );
        let ray = Ray(uniforms.camera_pos, direction);

        let sample_color = ray_color(ray);
        pixel_color = vec3_add(pixel_color, sample_color);
        color_sum = vec3_add(color_sum, sample_color);
        color_squared_sum = vec3_add(color_squared_sum, vec3_mul_vec3(sample_color, sample_color));
    }

    // Calculate variance and decide if more samples needed
    let mean = vec3_div(color_sum, f32(base_samples));
    let mean_squared = vec3_mul_vec3(mean, mean);
    let variance_vec = vec3_sub(vec3_div(color_squared_sum, f32(base_samples)), mean_squared);
    let variance_value = variance_vec.x + variance_vec.y + variance_vec.z; // Sum of RGB variances
    
    // Adaptive sampling: add more samples if variance is high
    if (variance_value > 1.0 && base_samples < max_samples) {
        let additional_samples = min(max_samples - base_samples, u32(variance_value * 50.0));
        
        for (var i = 0u; i < additional_samples; i = i + 1) {
            rand_state = pcg_hash(rand_state + (base_samples + i) * 5096u);

            let u = (f32(x) + random_f32()) / f32(uniforms.width - 1u);
            let v = (f32(uniforms.height - 1u - y) + random_f32()) / f32(uniforms.height - 1u);

            let half_height = tan(f32(uniforms.fov_rad) / 2.0);
            let half_width = f32(uniforms.aspect_ratio) * half_height;
            let viewport_x = half_width * (2.0 * u - 1.0);
            let viewport_y = half_height * (2.0 * v - 1.0) / f32(uniforms.char_aspect_ratio);

            let direction = normalize(
                vec3_add(
                    vec3_add(
                        uniforms.camera_forward,
                        vec3_mul(uniforms.camera_right, viewport_x)
                    ),
                    vec3_mul(uniforms.camera_up, viewport_y)
                )
            );
            let ray = Ray(uniforms.camera_pos, direction);

            pixel_color = vec3_add(pixel_color, ray_color(ray));
        }
        
        // Update total sample count for proper averaging
        let total_samples = base_samples + additional_samples;
        pixel_color = vec3_div(vec3_mul(pixel_color, f32(uniforms.samples_per_pixel)), f32(total_samples));
    }

    // Define index here, before using it
    let index = y * uniforms.width + x;
    
    // Store variance information for debugging/visualization
    variance[index] = VarianceBuffer(variance_value, base_samples);

    let current_sample = vec3_div(pixel_color, f32(uniforms.samples_per_pixel));

    if (uniforms.frame_number == 0u) {
        accumulation[index] = current_sample;
    } else {
        let alpha = 1.0 / f32(uniforms.frame_number + 1u);
        accumulation[index] = vec3_add(
            vec3_mul(accumulation[index], 1.0 - alpha),
            vec3_mul(current_sample, alpha)
        );
    }
    pixels[index] = accumulation[index];
}
