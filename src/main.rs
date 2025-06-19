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
    // Convert the pixel color to RGB values (0 to 255)
    let r = (pixel_color.x * 255.0).clamp(0.0, 255.0) as u8;
    let g = (pixel_color.y * 255.0).clamp(0.0, 255.0) as u8;
    let b = (pixel_color.z * 255.0).clamp(0.0, 255.0) as u8;

    let chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    let index = ((brightness * (chars.len() - 1) as f64) as usize).min(chars.len() - 1);
    print!("{}", chars[index]);
}
    
fn main() {
    // Set the dimensions of the image
    let width = 400;
    let height = 200;

    let aspect_ratio = width as f64 / height as f64;

    // Create a world
    let mut world = HittableList::new();
    
    // Add a center sphere
    world.add(Box::new(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5)));
    world.add(Box::new(Sphere::new(Vec3::new(-1.0, -0.1, -1.0), 0.2)));

    // Add a floor (large sphere)
    world.add(Box::new(Sphere::new(Vec3::new(0.0, -100.5, -1.0), 100.0)));


    for j in (0..height).rev() {
        for i in 0..width {
            let u = i as f64 / (width - 1) as f64;
            let v = j as f64 / (height - 1) as f64;

            let pixel_aspect = 2.0;
            let viewport_x = aspect_ratio * (2.0 * u - 1.0);
            let viewport_y = (2.0 * v - 1.0) * pixel_aspect;

            let direction = Vec3::new(viewport_x, viewport_y, -1.0).normalize();
            let ray = Ray {
                origin: Vec3::new(0.0, 0.0, 0.0),
                direction,
            };

            let black = Vec3::new(0.0, 0.0, 0.0);

            if let Some(hit_record) = world.hit(&ray, 0.001, f64::INFINITY) {
                let light_dir = Vec3::new(-1.0, 1.0, 1.0).normalize();

                let shadow_origin = hit_record.p + hit_record.normal * 1e-4;
                let shadow_ray = Ray {
                    origin: shadow_origin,
                    direction: light_dir,
                };

                let in_shadow = world.hit(&shadow_ray, 0.001, f64::INFINITY).is_some();

                let mut brightness = hit_record.normal.dot(&light_dir).max(0.0);
                if in_shadow {
                    brightness *= 0.2; // Dim if in shadow
                }

                let pixel_color = Vec3::new(brightness, brightness, brightness);
                write_color(pixel_color);
            } else {
                // If no hit, write black
                write_color(black);
            }
        }
        println!();
    }
}