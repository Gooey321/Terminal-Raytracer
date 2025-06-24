struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
};

struct Sphere {
    center: Vec3,
    radius: f32,
    color: Vec3,
    _padding0: f32,
    emission: Vec3,
    reflectivity: f32,
    roughness: f32,
    _p1: f32,  
}

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
    roughness: f32,
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
    lens_radius: f32,
    focal_distance: f32,
    _padding3: f32
};

const FRAMES_TO_ACCUMULATE: u32 = 16u;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(2) var<storage, read_write> pixels: array<Vec3>;
@group(0) @binding(3) var<storage, read_write> accumulation: array<Vec3>;

// Vector operations
fn vec3_add(a: Vec3, b: Vec3) -> Vec3 {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

fn vec3_sub(a: Vec3, b: Vec3) -> Vec3 {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

fn vec3_mul(a: Vec3, s: f32) -> Vec3 {
    return Vec3(a.x * s, a.y * s, a.z * s);
}

fn vec3_mul_vec3(a: Vec3, b: Vec3) -> Vec3 {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

fn vec3_div(a: Vec3, s: f32) -> Vec3 {
    return Vec3(a.x / s, a.y / s, a.z / s);
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
        let p = Vec3(random_f32() * 2.0 - 1.0, random_f32() * 2.0 - 1.0, random_f32() * 2.0 - 1.0);
        if (dot(p,p) < 1.0) { return p; }
    }
    // Fallback if we don't find a point in the sphere after 100 tries
    return Vec3(0.0, 1.0, 0.0);
}

fn random_in_unit_disk() -> Vec3 {
    for (var i = 0; i < 100; i = i + 1) {
        let p = Vec3(random_f32() * 2.0 - 1.0, random_f32() * 2.0 - 1.0, 0.0);
        if (dot(p, p) < 1.0) {
            return p;
        }
    }
    return Vec3(0.0, 0.0, 0.0);
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

// Ray tracing logic
fn hit_scene(ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
    var closest_so_far = t_max;
    var hit: HitRecord;
    hit.t = -1.0;

    for (var i = 0u; i < arrayLength(&spheres); i = i + 1) {
        let s = spheres[i];
        let oc = vec3_sub(s.center, ray.origin);
        let a = dot(ray.direction, ray.direction);
        let h = dot(ray.direction, oc);
        let c = dot(oc, oc) - s.radius * s.radius;
        let discriminant = h * h - a * c;

        if (discriminant >= 0.0) {
            let sqrtd = sqrt(discriminant);
            var root = (h - sqrtd) / a;
            if (root <= t_min || closest_so_far <= root) {
                root = (h + sqrtd) / a;
                if (root <= t_min || closest_so_far <= root) { continue; }
            }
            
            closest_so_far = root;
            hit.t = root;
            hit.p = vec3_add(ray.origin, vec3_mul(ray.direction, root));
            let outward_normal = vec3_div(vec3_sub(hit.p, s.center), s.radius);
            hit.front_face = dot(ray.direction, outward_normal) < 0.0;
            if (hit.front_face) {
                hit.normal = outward_normal;
            } else {
                hit.normal = vec3_mul(outward_normal, -1.0);
            }
            hit.color = s.color;
            hit.emission = s.emission;
            hit.reflectivity = s.reflectivity;
            hit.roughness = s.roughness;
        }
    }
    return hit;
}

fn sample_lights(hit_point: Vec3, normal: Vec3, surface_color: Vec3) -> Vec3 {
    var total_light = Vec3(0.0, 0.0, 0.0);
    
    // Iterate through all spheres to find emissive ones
    for (var i = 0u; i < arrayLength(&spheres); i = i + 1) {
        let light_sphere = spheres[i];
        
        // Check if this sphere is emissive (has emission > 0)
        let emission_strength = light_sphere.emission.x + light_sphere.emission.y + light_sphere.emission.z;
        if (emission_strength > 0.0) {
            // This is a light source - sample it
            
            // Sample random point on light sphere surface
            let to_light_center = vec3_sub(light_sphere.center, hit_point);
            let distance_to_center = length(to_light_center);
            
            // Skip if we're inside the light sphere
            if (distance_to_center < light_sphere.radius) {
                continue;
            }
            
            // Generate random point on sphere surface
            let random_dir = random_in_unit_sphere();
            let light_surface_point = vec3_add(light_sphere.center, vec3_mul(normalize(random_dir), light_sphere.radius));
            
            let light_direction = vec3_sub(light_surface_point, hit_point);
            let light_distance = length(light_direction);
            let light_dir_normalized = normalize(light_direction);
            
            // Check if light is above surface (don't light from below)
            let cos_theta = dot(normal, light_dir_normalized);
            if (cos_theta <= 0.0) {
                continue;
            }
            
            // Cast shadow ray
            let shadow_ray = Ray(vec3_add(hit_point, vec3_mul(normal, 0.001)), light_dir_normalized);
            let shadow_hit = hit_scene(shadow_ray, 0.001, light_distance - 0.001);
            
            // If no obstruction, add light contribution
            if (shadow_hit.t < 0.0) {
                // Calculate light falloff (inverse square law)
                let attenuation = 1.0 / (light_distance * light_distance);
                
                // Calculate surface area of light (for proper energy conservation)
                let light_area = 4.0 * 3.14159 * light_sphere.radius * light_sphere.radius;
                
                // Add the light contribution
                let light_contrib = vec3_mul_vec3(
                    vec3_mul_vec3(light_sphere.emission, surface_color),
                    Vec3(cos_theta * attenuation * light_area, 
                         cos_theta * attenuation * light_area,
                         cos_theta * attenuation * light_area)
                );
                
                total_light = vec3_add(total_light, light_contrib);
            }
        }
    }
    
    return total_light;
}

fn ray_color(initial_ray: Ray) -> Vec3 {
    var accumulated_color = Vec3(0.0, 0.0, 0.0);
    var attenuation = Vec3(1.0, 1.0, 1.0);
    var current_ray = initial_ray;

    for (var i = 0u; i < uniforms.max_depth; i = i + 1) {
        let hit = hit_scene(current_ray, 0.001, 1e10);

        if (hit.t < 0.0) {
            // Sky color
            accumulated_color = vec3_add(accumulated_color, vec3_mul_vec3(Vec3(0.05, 0.05, 0.1), attenuation));
            break;
        }

        // Add emission if this surface is emissive
        accumulated_color = vec3_add(accumulated_color, vec3_mul_vec3(hit.emission, attenuation));
        
        // Add direct lighting from all light sources (only on first bounce for performance)
        if (true) {
            let direct_light = sample_lights(hit.p, hit.normal, hit.color);
            accumulated_color = vec3_add(accumulated_color, vec3_mul_vec3(direct_light, attenuation));
        }
        
        // Update attenuation for next bounce
        attenuation = vec3_mul_vec3(attenuation, hit.color);
        
        // Russian roulette termination
        let survival_prob = max(attenuation.x, max(attenuation.y, attenuation.z));
        if (i > 1u && survival_prob < random_f32() * 0.8) {
            break;
        }
        attenuation = vec3_div(attenuation, survival_prob);

        let f0 = hit.reflectivity;
        let cosine = abs(dot(current_ray.direction, hit.normal));
        let fresnel_reflectance = f0 + (1.0 - f0) * pow(1.0 - cosine, 5.0);

        if (random_f32() < fresnel_reflectance) {
            // Reflective surface
            let reflected_dir = reflect(current_ray.direction, hit.normal);
            let scattered = normalize(vec3_add(reflected_dir, vec3_mul(random_in_unit_sphere(), hit.roughness)));
            current_ray = Ray(hit.p, scattered);
        } else {
            // Diffuse surface
            let scatter_direction = normalize(vec3_add(hit.normal, random_in_unit_sphere()));
            current_ray = Ray(hit.p, scatter_direction);
        }
    }
    return accumulated_color;
}

// Bilateral filter for denoising
fn bilateral_filter(center_pos: vec2<u32>) -> Vec3 {
    let center_color = pixels[center_pos.y * uniforms.width + center_pos.x];
    let center_brightness = dot(center_color, Vec3(0.299, 0.587, 0.114));
    
    var sum = Vec3(0.0, 0.0, 0.0);
    var weight_sum = 0.0;
    let radius = 2;
    
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let sample_x = i32(center_pos.x) + dx;
            let sample_y = i32(center_pos.y) + dy;
            
            if (sample_x >= 0 && sample_x < i32(uniforms.width) && 
                sample_y >= 0 && sample_y < i32(uniforms.height)) {
                
                let sample_pos = vec2<u32>(u32(sample_x), u32(sample_y));
                let sample_color = pixels[sample_pos.y * uniforms.width + sample_pos.x];
                let sample_brightness = dot(sample_color, Vec3(0.299, 0.587, 0.114));
                
                // Spatial weight (Gaussian-like)
                let spatial_dist = f32(dx * dx + dy * dy);
                let spatial_weight = exp(-spatial_dist / 8.0);
                
                // Bilateral weight (preserve edges)
                let brightness_diff = abs(center_brightness - sample_brightness);
                let bilateral_weight = exp(-brightness_diff * 10.0);
                
                let total_weight = spatial_weight * bilateral_weight;
                sum = vec3_add(sum, vec3_mul(sample_color, total_weight));
                weight_sum = weight_sum + total_weight;
            }
        }
    }
    
    if (weight_sum > 0.0) {
        return vec3_div(sum, weight_sum);
    } else {
        return center_color;
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }
    
    rand_state = (y * uniforms.width + x) * 1973u + uniforms.seed * 9277u + uniforms.frame_number * 12345u;
    
    var pixel_color = Vec3(0.0, 0.0, 0.0);
    for (var i = 0u; i < uniforms.samples_per_pixel; i = i + 1) {
        rand_state = pcg_hash(rand_state + i * 5096u);

        let u = (f32(x) + random_f32()) / f32(uniforms.width - 1u);
        let v = (f32(uniforms.height - 1u - y) + random_f32()) / f32(uniforms.height - 1u);

        let half_height = tan(f32(uniforms.fov_rad) / 2.0);
        let half_width = f32(uniforms.aspect_ratio) * half_height;
        let viewport_x = half_width * (2.0 * u - 1.0);
        let viewport_y = half_height * (2.0 * v - 1.0) / f32(uniforms.char_aspect_ratio);

        let direction = normalize(Vec3(viewport_x, viewport_y, -1.0));
        
        // Add depth of field
        let origin = Vec3(0.0, 0.0, 0.0);
        let rd = vec3_mul(random_in_unit_disk(), uniforms.lens_radius);
        let offset = Vec3(rd.x, rd.y, 0.0);
        let ray_origin = vec3_add(origin, offset);
        let focal_point = vec3_add(origin, vec3_mul(direction, uniforms.focal_distance));
        let ray_direction = normalize(vec3_sub(focal_point, ray_origin));
        let ray = Ray(ray_origin, ray_direction);

        pixel_color = vec3_add(pixel_color, ray_color(ray));
    }

    let index = y * uniforms.width + x;
    let current_sample = vec3_div(pixel_color, f32(uniforms.samples_per_pixel));

    // Store raw result first
    if (uniforms.frame_number == 0u) {
        accumulation[index] = current_sample;
    } else {
        let alpha = 1.0 / f32(uniforms.frame_number + 1u);
        accumulation[index] = vec3_add(
            vec3_mul(accumulation[index], 1.0 - alpha),
            vec3_mul(current_sample, alpha)
        );
    }
    
    // Copy the latest accumulated value to the pixel buffer for display/denoising
    pixels[index] = accumulation[index];

    // Apply denoising on the final frame
    if (uniforms.frame_number == (FRAMES_TO_ACCUMULATE - 1u)) {
        // Ensure all threads in the workgroup have written their accumulated pixel
        // before we start reading neighbor pixels for the filter.
        workgroupBarrier();

        let denoised = bilateral_filter(vec2<u32>(x, y));
        pixels[index] = denoised;
    }
}
