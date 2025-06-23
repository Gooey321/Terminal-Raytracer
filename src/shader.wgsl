struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
};

struct Sphere {
    center: Vec3,
    radius: f32,
    color: Vec3,
    emission: Vec3,
    reflectivity: f32,
    _p1: f32,
    _p2: f32,  
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
    _padding3: f32
};

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
        }
    }
    return hit;
}

fn ray_color(initial_ray: Ray) -> Vec3 {
    var accumulated_color = Vec3(0.0, 0.0, 0.0);
    var attenuation = Vec3(1.0, 1.0, 1.0);
    var current_ray = initial_ray;

    for (var i = 0u; i < uniforms.max_depth; i = i + 1) {
        if (i > 2u) {
            let survival_prob = max(attenuation.x, max(attenuation.y, attenuation.z));
            if (survival_prob < random_f32()) {
                break;
            }
            attenuation = vec3_div(attenuation, survival_prob);
        }
        let hit = hit_scene(current_ray, 0.001, 1e10);

        if (hit.t < 0.0) {
            accumulated_color = vec3_add(accumulated_color, vec3_mul_vec3(Vec3(0.0, 0.0, 0.0), attenuation));
            break;
        }

        accumulated_color = vec3_add(accumulated_color, vec3_mul_vec3(hit.emission, attenuation));
        attenuation = vec3_mul_vec3(attenuation, hit.color);

        let is_reflective = hit.reflectivity > random_f32();
        if (is_reflective) {
            let reflected_dir = reflect(current_ray.direction, hit.normal);
            current_ray = Ray(hit.p, reflected_dir);
        } else {
            let scatter_direction = normalize(vec3_add(hit.normal, random_in_unit_sphere()));
            current_ray = Ray(hit.p, scatter_direction);
        }
    }
    return accumulated_color;
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
        let ray = Ray(Vec3(0.0, 0.0, 0.0), direction);

        pixel_color = vec3_add(pixel_color, ray_color(ray));
    }

    let index = y * uniforms.width + x;
    let current_sample = vec3_div(pixel_color, f32(uniforms.samples_per_pixel));

    if (uniforms.frame_number == 0u) {
        accumulation[index] = current_sample;
        pixels[index] = current_sample;
    } else {
        let alpha = 1.0 / f32(uniforms.frame_number + 1u);
        accumulation[index] = vec3_add(
            vec3_mul(accumulation[index], 1.0 - alpha),
            vec3_mul(current_sample, alpha)
        );
    }
    pixels[index] = accumulation[index];
}
