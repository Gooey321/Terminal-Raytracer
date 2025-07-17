// src/sphere.rs
use crate::vec3::Vec3;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub _padding1: f32,
    pub _padding2: f32,
    pub _padding3: f32,
    pub color: Vec3,
    pub emission: Vec3,
    pub reflectivity: f32,
    pub _padding4: f32,
    pub _padding5: f32,
    pub _padding6: f32,
}

// Function to create a new Sphere
impl Sphere {
    pub fn new(center: Vec3, radius: f64, color: Vec3, emission: Vec3, reflectivity: f64) -> Self {
        Self {
            center,
            radius: (radius.max(0.0)) as f32,
            _padding1: 0.0,
            _padding2: 0.0,
            _padding3: 0.0,
            color,
            emission,
            reflectivity: reflectivity as f32,
            _padding4: 0.0,
            _padding5: 0.0,
            _padding6: 0.0,
        }
    }
}