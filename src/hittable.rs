// src/hittable.rs
use crate::vec3::Vec3;
use crate::ray::Ray;

#[derive(Debug, Clone)]
pub struct HitRecord {
    pub p: Vec3,
    pub normal: Vec3,
    pub t: f64,
    pub front_face: bool,
    pub color: Vec3,
    pub emission: Vec3,
    pub reflectivity: f64,
}

impl HitRecord {
    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3) {
        self.front_face = ray.direction.dot(&outward_normal) < 0.0;
        self.normal = if self.front_face { 
            outward_normal 
        } else { 
            outward_normal * -1.0 
        };
    }
}
pub trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
}