// src/vec3.rs
use std::ops::{Add, Sub, Mul, Div};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _padding: f32,  // Add padding to make Vec3 16 bytes
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x: x as f32, y: y as f32, z: z as f32, _padding: 0.0 }
    }

    // Add a new function for direct f32 construction
    pub fn new_f32(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z, _padding: 0.0 }
    }

    pub fn cross(self, rhs: Vec3) -> Vec3 {
        Vec3::new_f32(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x
        )
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new_f32(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z
        )
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new_f32(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z
        )
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Vec3 {
        Vec3::new_f32(
            self.x * rhs,
            self.y * rhs,
            self.z * rhs
        )
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3::new_f32(
            self.x * rhs.x,
            self.y * rhs.y,
            self.z * rhs.z
        )
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f32) -> Vec3 {
        Vec3::new_f32(
            self.x / rhs,
            self.y / rhs,
            self.z / rhs
        )
    }
}
