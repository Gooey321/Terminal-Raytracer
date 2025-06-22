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

const MAX_DEPTH: i32 = 10; // Maximum recursion depth for reflections
const SAMPLES_PER_PIXEL: i32 = 128; // Use a moderate number of samples for a good balance

fn ray_color(ray: &Ray, world: &HittableList, depth: i32) -> Vec3 {

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if depth <= 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    if let Some(hit_record) = world.hit(ray, 0.001, f64::INFINITY) {
        let emission = hit_record.emission;
        let shadow_samples = 4;
        let mut visibility = 0.0;

        // light values, hardcoded for now
        let light_center = Vec3::new(0.0, 0.8, -3.0);
        let light_radius = 0.3;
        let light_emission = Vec3::new(1.0, 1.0, 1.0);

        let mut rng = rand::thread_rng();

        // Calculate direct lighting
        for _ in 0..shadow_samples {
            // generate a random point on the light source
            let rand_vec = loop {
                let p = Vec3::new(
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                );
                if p.length_squared() < 1.0 {
                    break p;
                }
            };
            let rand_light_point = light_center + rand_vec.normalize() * light_radius;

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
        let direct_light = light_emission * average_visibility;

        let ambient_light = Vec3::new(1.0, 1.0, 1.0) * 0.2;
        let surface_color = hit_record.color * (direct_light + ambient_light);

        // Reflection
        let reflected_ray = Ray {
            origin: hit_record.p + hit_record.normal * 1e-4,
            direction: ray.direction.reflect(&hit_record.normal),
        };
        let reflected_color = ray_color(&reflected_ray, world, depth - 1);

        let final_color = surface_color * (1.0 - hit_record.reflectivity) + reflected_color * hit_record.reflectivity;

        return emission + final_color;
    }

    // If no hit, return black
    Vec3::new(0.0, 0.0, 0.0)
}


fn write_color(pixel_color: Vec3, full_color: bool) {
    let r = (pixel_color.x.sqrt() * 255.0).clamp(0.0, 255.0) as u8;
    let g = (pixel_color.y.sqrt() * 255.0).clamp(0.0, 255.0) as u8;
    let b = (pixel_color.z.sqrt() * 255.0).clamp(0.0, 255.0) as u8;

    let brightness = 0.2126 * pixel_color.x.sqrt() + 0.7152 * pixel_color.y.sqrt() + 0.0722 * pixel_color.z.sqrt();

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
    
    // Add a ceiling light
    world.add(Box::new(Sphere::new_emissive(
        Vec3::new(0.0, 0.8, -3.0),
        0.3,
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        0.0
    )));

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
                pixel_color = pixel_color + ray_color(&ray, &world, MAX_DEPTH);
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