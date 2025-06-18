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

fn main() {
    let width = 200;
    let height = 100;

    let aspect_ratio = width as f64 / height as f64;

    // Create a world with multiple objects
    let mut world = HittableList::new();
    
    // Add a center sphere
    world.add(Box::new(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5)));
    
    // Add a ground sphere (large sphere below)
    world.add(Box::new(Sphere::new(Vec3::new(0.0, -100.5, -1.0), 100.0)));
    
    // Add a few more spheres for interest
    world.add(Box::new(Sphere::new(Vec3::new(-1.0, 0.0, -1.0), 0.5)));
    // world.add(Box::new(Sphere::new(Vec3::new(1.0, 0.0, -1.0), 0.5)));

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

            // Now hit against the entire world
            if let Some(hit_record) = world.hit(&ray, 0.001, f64::INFINITY) {
                let light_dir = Vec3::new(-1.0, 1.0, 1.0).normalize();
                let brightness = hit_record.normal.dot(&light_dir).max(0.0);
                let chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
                let index = ((brightness * (chars.len() - 1) as f64) as usize).min(chars.len() - 1);
                print!("{}", chars[index]);
            } else {
                print!(" ");
            }
        }
        println!();
    }
}