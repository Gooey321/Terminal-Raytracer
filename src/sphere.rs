// src/sphere.rs
use crate::vec3::Vec3;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub color: Vec3, // Colour of the sphere.
    pub emission: Vec3,
    pub reflectivity: f32,

    _p1: f32,
    _p2: f32,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f64, color: Vec3, emission: Vec3, reflectivity: f64) -> Self {
        Self {
            center,
            radius: (radius.max(0.0)) as f32,
            color,
            emission,
            reflectivity: reflectivity as f32,
            _p1: 0.0,
            _p2: 0.0,
        }
    }
}