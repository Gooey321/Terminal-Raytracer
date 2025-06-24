// src/sphere.rs
use crate::vec3::Vec3;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub color: Vec3, // Colour of the sphere.
    pub _padding0: f32, // Padding to ensure correct alignment for `emission`
    pub emission: Vec3,
    pub reflectivity: f32,
    pub roughness: f32, // New field for glossy reflections

    _p1: f32,
    _p2: [f32; 2],
}

impl Sphere {
    pub fn new(center: Vec3, radius: f64, color: Vec3, emission: Vec3, reflectivity: f64, roughness: f64) -> Self {
        Self {
            center,
            radius: (radius.max(0.0)) as f32,
            color,
            _padding0: 0.0,
            emission,
            reflectivity: reflectivity as f32,
            roughness: roughness as f32,
            _p1: 0.0,
            _p2: [0.0; 2],
        }
    }
}