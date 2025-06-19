// src/sphere.rs
use crate::vec3::Vec3;
use crate::ray::Ray;
use crate::hittable::{HitRecord, Hittable};

pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub color: Vec3, // Colour of the sphere.
}

impl Sphere {
    pub fn new(center: Vec3, radius: f64, color: Vec3) -> Self {
        Self {
            center,
            radius: radius.max(0.0),
            color,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = self.center - ray.origin;
        let a = ray.direction.length_squared();
        let h = ray.direction.dot(&oc);
        let c = oc.length_squared() - self.radius * self.radius;
        
        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();

        // Find the nearest root that lies in the acceptable range
        let mut root = (h - sqrtd) / a;
        if root <= t_min || t_max <= root {
            root = (h + sqrtd) / a;
            if root <= t_min || t_max <= root {
                return None;
            }
        }

        let t = root;
        let p = ray.origin + ray.direction * t;
        let outward_normal = (p - self.center) / self.radius;
        
        let mut hit_record = HitRecord {
            p,
            normal: outward_normal,
            t,
            front_face: false,
            color: self.color,
        };

        hit_record.set_face_normal(ray, outward_normal);

        Some(hit_record)
    }
}
