use crate::vec3::Vec3;

// Camera struct to hold position and orientation
pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    // Create a new camera with a position and orientation
    pub fn new(position: Vec3, yaw: f32, pitch: f32) -> Self {
        Self { position, yaw, pitch }
    }

    // Calculate forward, right, and up vectors for the camera
    pub fn calculate_vectors(&self) -> (Vec3, Vec3, Vec3) {
        let forward = Vec3::new_f32(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        );
        let right = Vec3::new_f32(-self.yaw.sin(), 0.0, self.yaw.cos());
        let up = right.cross(forward);
        (forward, right, up)
    }
}