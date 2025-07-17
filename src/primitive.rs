use crate::vec3::Vec3;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Primitive {
    pub primitive_type: u32, // 0 = sphere, 1 = plane, 2 = triangle
    pub _padding1: u32,
    pub _padding2: u32,
    pub _padding3: u32,
    
    // Material properties
    pub color: Vec3,
    pub emission: Vec3,
    pub reflectivity: f32,
    pub _padding4: f32,
    pub _padding5: f32,
    pub _padding6: f32,
    
    // Sphere data (when primitive_type == 0)
    pub sphere_center: Vec3,
    pub sphere_radius: f32,
    pub _sphere_padding: [f32; 3],
    
    // Plane data (when primitive_type == 1)
    pub plane_point: Vec3,
    pub plane_normal: Vec3,
    
    // Triangle data (when primitive_type == 2)
    pub triangle_v0: Vec3,
    pub triangle_v1: Vec3,
    pub triangle_v2: Vec3,
    pub _triangle_padding: f32,
}

impl Primitive {
    pub fn new_sphere(center: Vec3, radius: f64, color: Vec3, emission: Vec3, reflectivity: f64) -> Self {
        Self {
            primitive_type: 0,
            _padding1: 0, _padding2: 0, _padding3: 0,
            color, emission,
            reflectivity: reflectivity as f32,
            _padding4: 0.0, _padding5: 0.0, _padding6: 0.0,
            sphere_center: center,
            sphere_radius: radius as f32,
            _sphere_padding: [0.0; 3],
            plane_point: Vec3::new_f32(0.0, 0.0, 0.0),
            plane_normal: Vec3::new_f32(0.0, 0.0, 0.0),
            triangle_v0: Vec3::new_f32(0.0, 0.0, 0.0),
            triangle_v1: Vec3::new_f32(0.0, 0.0, 0.0),
            triangle_v2: Vec3::new_f32(0.0, 0.0, 0.0),
            _triangle_padding: 0.0,
        }
    }
    
    pub fn new_plane(point: Vec3, normal: Vec3, color: Vec3, emission: Vec3, reflectivity: f64) -> Self {
        Self {
            primitive_type: 1,
            _padding1: 0, _padding2: 0, _padding3: 0,
            color, emission,
            reflectivity: reflectivity as f32,
            _padding4: 0.0, _padding5: 0.0, _padding6: 0.0,
            sphere_center: Vec3::new_f32(0.0, 0.0, 0.0),
            sphere_radius: 0.0,
            _sphere_padding: [0.0; 3],
            plane_point: point,
            plane_normal: normal,
            triangle_v0: Vec3::new_f32(0.0, 0.0, 0.0),
            triangle_v1: Vec3::new_f32(0.0, 0.0, 0.0),
            triangle_v2: Vec3::new_f32(0.0, 0.0, 0.0),
            _triangle_padding: 0.0,
        }
    }
    
    pub fn new_triangle(v0: Vec3, v1: Vec3, v2: Vec3, color: Vec3, emission: Vec3, reflectivity: f64) -> Self {
        Self {
            primitive_type: 2,
            _padding1: 0, _padding2: 0, _padding3: 0,
            color, emission,
            reflectivity: reflectivity as f32,
            _padding4: 0.0, _padding5: 0.0, _padding6: 0.0,
            sphere_center: Vec3::new_f32(0.0, 0.0, 0.0),
            sphere_radius: 0.0,
            _sphere_padding: [0.0; 3],
            plane_point: Vec3::new_f32(0.0, 0.0, 0.0),
            plane_normal: Vec3::new_f32(0.0, 0.0, 0.0),
            triangle_v0: v0,
            triangle_v1: v1,
            triangle_v2: v2,
            _triangle_padding: 0.0,
        }
    }
}