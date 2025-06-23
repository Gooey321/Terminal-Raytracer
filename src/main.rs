mod vec3;
mod ray;
mod sphere;
mod hittable;
mod hittable_list;

use core::f64;
use std::time::Instant;
use vec3::Vec3;
use ray::Ray;
use sphere::Sphere;
use hittable::Hittable;
use hittable_list::HittableList;
use rand::Rng;
use rayon::prelude::*;

const MAX_DEPTH: i32 = 8; // Maximum recursion depth for reflections
const SAMPLES_PER_PIXEL: i32 = 256; // Use a moderate number of samples for a good balance

#[derive(Clone, Copy)]
struct Light {
    center: Vec3,
    radius: f64,
    emission: Vec3,
}

impl Light {
    fn new(center: Vec3, radius: f64, emission: Vec3) -> Self {
        Light { center, radius, emission }
    }
}

fn ray_color(ray: &Ray, world: &HittableList, depth: i32, lights: &[Light]) -> Vec3 {
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if depth <= 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    if let Some(hit_record) = world.hit(ray, 0.001, f64::INFINITY) {
        let emission = hit_record.emission;
        let mut total_direct_light = Vec3::new(0.0, 0.0, 0.0);
        let mut rng = rand::thread_rng(); // Move RNG creation here

        // Calculate lighting from ALL lights in the list
        for light in lights {
            let shadow_samples = 64; // Increased for cleaner shadows
            let mut visibility = 0.0;

            // Calculate direct lighting for this specific light
            for _ in 0..shadow_samples {
                let u1 = rng.r#gen::<f64>();
                let u2 = rng.r#gen::<f64>();

                let z = 1.0 - 2.0 * u1;
                let r = (1.0 - z * z).sqrt();
                let phi = 2.0 * std::f64::consts::PI * u2;


                // Generate a random point on this light's surface
                let rand_vec = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                let rand_light_point = light.center + rand_vec * light.radius;

                // Cast a shadow ray towards that random point
                let shadow_ray_orig = hit_record.p + hit_record.normal * 1e-4;
                let light_dir = rand_light_point - shadow_ray_orig;
                let distance_light = light_dir.length();

                let shadow_ray = Ray {
                    origin: shadow_ray_orig,
                    direction: light_dir.normalize(),
                };

                // Check if the path is obstructed
                if world.hit(&shadow_ray, 0.001, distance_light - 0.001).is_none() {
                    visibility += 1.0;
                }
            }

            let average_visibility = visibility / shadow_samples as f64;
            // Add this light's contribution to the total
            total_direct_light = total_direct_light + light.emission * average_visibility;
        }

        let ambient_light = Vec3::new(1.0, 1.0, 1.0) * 0.1; // Reduced for better contrast
        let surface_color = hit_record.color * (total_direct_light + ambient_light);

        // Only calculate reflections if reflectivity is significant
        let reflected_color = if hit_record.reflectivity > 0.01 {
            let reflected_ray = Ray {
                origin: hit_record.p + hit_record.normal * 1e-4,
                direction: ray.direction.reflect(&hit_record.normal),
            };
            ray_color(&reflected_ray, world, depth - 1, lights) * hit_record.reflectivity
        } else {
            Vec3::new(0.0, 0.0, 0.0)
        };

        let final_color = surface_color * (1.0 - hit_record.reflectivity) + reflected_color;

        return emission + final_color;
    }

    // If no hit, return black
    Vec3::new(0.0, 0.0, 0.0)
}


fn write_color(pixel_color: Vec3, full_color: bool) {
    let r = (pixel_color.x * 255.0).clamp(0.0, 255.0) as u8;
    let g = (pixel_color.y * 255.0).clamp(0.0, 255.0) as u8;
    let b = (pixel_color.z * 255.0).clamp(0.0, 255.0) as u8;

    let brightness = 0.2126 * pixel_color.x + 0.7152 * pixel_color.y + 0.0722 * pixel_color.z;

    if full_color {
        print!("\x1b[38;2;{};{};{}m█\x1b[0m", r, g, b);
    } else {
        let chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
        let index = ((brightness * (chars.len() - 1) as f64) as usize).min(chars.len() - 1);

        // Print the color with ANSI
        print!("\x1b[38;2;{};{};{}m{}\x1b[0m", r, g, b, chars[index]);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let full_color;
    if args.contains(&"--full-color".to_string()) {
        println!("outputting with █ characters");
        full_color = true;
    } else {
        full_color = false;
        println!("outputting with ASCII characters");
    }

    // Set the dimensions of the image
    let width = 200;
    let height = 100;

    let aspect_ratio = width as f64 / height as f64;

    // Create a world
    let mut world = HittableList::new();

    world.add(Box::new(Sphere::new(Vec3::new(-1001.0, 0.0, -3.0), 1000.0, Vec3::new(0.75, 0.25, 0.25), 0.0)));  
    world.add(Box::new(Sphere::new(Vec3::new(1001.0, 0.0, -3.0), 1000.0, Vec3::new(0.25, 0.75, 0.25), 0.0)));
    world.add(Box::new(Sphere::new(Vec3::new(0.0, 0.0, -1004.0), 1000.0, Vec3::new(0.75, 0.75, 0.75), 0.0)));
    world.add(Box::new(Sphere::new(Vec3::new(0.0, -1001.0, -3.0), 1000.0, Vec3::new(0.75, 0.75, 0.75), 0.0)));
    world.add(Box::new(Sphere::new(Vec3::new(0.0, 1001.0, -3.0), 1000.0, Vec3::new(0.75, 0.75, 0.75), 0.0)));
    world.add(Box::new(Sphere::new(Vec3::new(-0.5, -0.4, -2.5), 0.6, Vec3::new(0.9, 0.9, 0.9), 0.8)));  
    world.add(Box::new(Sphere::new(Vec3::new(0.5, -0.7, -3.2), 0.3, Vec3::new(0.6, 0.8, 0.9), 0.2)));
    
    let lights = vec![
        Light::new(Vec3::new(0.0, 0.8, -3.0), 0.3, Vec3::new(2.0, 2.0, 2.0)),
    ];

    for light in &lights {
        world.add(Box::new(Sphere::new_emissive(
            light.center,
            light.radius,
            Vec3::new(2.0, 2.0, 2.0),
            light.emission, 0.0)));
    }

    let start = Instant::now();
    println!("Starting render...");

    let pixels: Vec<(u32, u32)> = (0..height)
        .flat_map(|j| (0..width).map(move |i| (j, i)))
        .collect();

    let colors: Vec<Vec3> = pixels
        .par_iter()
        .map(|(j, i)| {
            let mut rng = rand::thread_rng();
            let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);

            // Start the multi-sampling loop for anti-aliasing
            for _ in 0..SAMPLES_PER_PIXEL {
                // Get a random point within the pixel
                let u = (*i as f64 + rng.r#gen::<f64>()) / (width - 1) as f64;
                let v = ((height - 1 - *j) as f64 + rng.r#gen::<f64>()) / (height - 1) as f64;

                let fov = 45.0_f64.to_radians();
                let half_height = (fov / 2.0).tan();
                let half_width = aspect_ratio * half_height;

                let viewport_x = half_width * (2.0 * u - 1.0);
                let viewport_y = half_height * (2.0 * v - 1.0);

                let char_aspect_ratio = 0.55;
                let corrected_viewport_y = viewport_y / char_aspect_ratio;

                let direction = Vec3::new(viewport_x, corrected_viewport_y, -1.0).normalize();
                let ray = Ray {
                    origin: Vec3::new(0.0, 0.0, 0.0),
                    direction,
                };
                
                // Accumulate the color from each sample
                pixel_color = pixel_color + ray_color(&ray, &world, MAX_DEPTH, &lights);
            }
            
            // Return the average color
            pixel_color / SAMPLES_PER_PIXEL as f64
        })
        .collect();

    // For every pixel
    for j in 0..height {
        for i in 0..width {
            let index = (j * width + i) as usize;
            write_color(colors[index], full_color);
        }
        println!();
    }

    let duration = start.elapsed();
    println!("Render completed in: {:?}", duration);
    println!("Image dimensions: {}x{}", width, height);
    println!("Total pixels: {}", width * height);
    println!("Average time per pixel: {:?}", duration / (width * height) as u32);
}