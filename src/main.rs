mod vec3;
mod ray;
mod sphere;
mod hittable;
mod hittable_list;

use vec3::Vec3;
use ray::Ray;
use sphere::Sphere;
use hittable::Hittable;
use hittable_list::HittableList;

fn write_color(pixel_color: Vec3) {
    let r = (pixel_color.x * 255.0).clamp(0.0, 255.0) as u8;
    let g = (pixel_color.y * 255.0).clamp(0.0, 255.0) as u8;
    let b = (pixel_color.z * 255.0).clamp(0.0, 255.0) as u8;

    let brightness = (0.2126 * r as f64 + 0.7152 * g as f64 + 0.0722 * b as f64) / 255.0;

    let chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    let index = ((brightness * (chars.len() - 1) as f64) as usize).min(chars.len() - 1);

    // Print the color with ANSI
    print!("\x1b[38;2;{};{};{}m{}\x1b[0m", r, g, b, chars[index]);
}

fn main() {
    // Set the dimensions of the image
    let width = 200;
    let height = 100;

    let aspect_ratio = width as f64 / height as f64;

    // Create a world
    let mut world = HittableList::new();

    // Left wall (red)
    world.add(Box::new(Sphere::new(Vec3::new(-1001.0, 0.0, -3.0), 1000.0, Vec3::new(0.75, 0.25, 0.25))));

    // Right wall (green)  
    world.add(Box::new(Sphere::new(Vec3::new(1001.0, 0.0, -3.0), 1000.0, Vec3::new(0.25, 0.75, 0.25))));

    // Back wall (white)
    world.add(Box::new(Sphere::new(Vec3::new(0.0, 0.0, -1004.0), 1000.0, Vec3::new(0.75, 0.75, 0.75))));

    // Floor (white)
    world.add(Box::new(Sphere::new(Vec3::new(0.0, -1001.0, -3.0), 1000.0, Vec3::new(0.75, 0.75, 0.75))));

    // Ceiling (white)
    world.add(Box::new(Sphere::new(Vec3::new(0.0, 1001.0, -3.0), 1000.0, Vec3::new(0.75, 0.75, 0.75))));

    // Two boxes (represented as spheres for now)

    // Left box (tall, white)
    world.add(Box::new(Sphere::new(Vec3::new(-0.5, -0.4, -2.5), 0.6, Vec3::new(0.75, 0.75, 0.75))));

    // Right box (short, white)  
    world.add(Box::new(Sphere::new(Vec3::new(0.5, -0.7, -3.2), 0.3, Vec3::new(0.75, 0.75, 0.75))));
    
    // Add a ceiling light (area light simulation)
    world.add(Box::new(Sphere::new_emissive(
        Vec3::new(0.0, 0.8, -3.0),    // Position: near ceiling
        0.3,                           // Small radius
        Vec3::new(1.0, 1.0, 1.0),     // White color
        Vec3::new(3.0, 3.0, 3.0)      // Bright white emission
    )));

    for j in (0..height).rev() {
        for i in 0..width {
            let u = i as f64 / (width - 1) as f64;
            let v = j as f64 / (height - 1) as f64;

            let fov = 45.0_f64.to_radians();
            let half_height = (fov / 2.0).tan();
            let half_width = aspect_ratio * half_height;

            let viewport_x = half_width * (2.0 * u - 1.0);
            let viewport_y = half_height * (2.0 * v - 1.0);

            let char_aspect_ratio = 0.6;
            let corrected_viewport_y = viewport_y / char_aspect_ratio;

            let direction = Vec3::new(viewport_x, corrected_viewport_y, -1.0).normalize();
            let ray = Ray {
                origin: Vec3::new(0.0, 0.0, 0.0),
                direction,
            };

            if let Some(hit_record) = world.hit(&ray, 0.001, f64::INFINITY) {
                let pixel_color = hit_record.emission;
                
                let light_pos = Vec3::new(0.0, 0.8, -3.0); 
                let light_emission = Vec3::new(3.0, 3.0, 3.0); // Emission color of the light source

                let light_dir = (light_pos - hit_record.p).normalize();
                let distance_to_light = (light_pos - hit_record.p).length();

                let shadow_ray = Ray {
                    origin: hit_record.p + hit_record.normal * 1e-4,
                    direction: light_dir,
                };

                let mut in_shadow = false;
                if let Some(shadow_hit) = world.hit(&shadow_ray, 0.001, distance_to_light - 0.001) {
                    if shadow_hit.emission.length_squared() < 0.01 {
                        in_shadow = true;
                    }
                }

                let mut direct_light = Vec3::new(0.0, 0.0, 0.0);
                if !in_shadow {
                    let brightness = hit_record.normal.dot(&light_dir).max(0.0);
                    let attenuation = 1.0 / (1.0 + distance_to_light * distance_to_light * 0.05);
                    direct_light = light_emission * brightness * attenuation;
                }

                let ambient_light = Vec3::new(1.0, 1.0, 1.0) * 0.5;

                let surface_color = hit_record.color * (direct_light + ambient_light);
                let final_color = pixel_color + surface_color;

                write_color(final_color);
            } else {
                // If no hit, write black
                write_color(Vec3::new(0.0, 0.0, 0.0));
            }
        }
        println!();
    }
}