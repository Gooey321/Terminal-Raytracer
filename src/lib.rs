mod vec3;
mod primitive;

use std::cell;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use vec3::Vec3;
use primitive::Primitive; // Add this import
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use serde::{Serialize, Deserialize};
use crossterm::{
    event::{self, Event, KeyCode},
    terminal,
    cursor,
    execute,
};
use std::io::{self, Write};

mod camera;
use camera::Camera;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    max_depth: u32,

    seed: u32,
    frame_number: u32,
    primitive_count: u32, // ADDED
    _padding1: u32,       // keep 16-byte alignment

    aspect_ratio: f32,
    char_aspect_ratio: f32,
    fov_rad: f32,
    _padding2: f32, // pad to 16

    camera_pos: Vec3,
    camera_forward: Vec3,
    camera_right: Vec3,
    camera_up: Vec3,

    grid_min: Vec3,        // ADDED
    inv_cell_size: Vec3,   // ADDED
    grid_dims: [u32; 3],   // ADDED: x,y,z
    _padding3: u32,        // pad to 16 bytes
}

#[derive(Serialize, Deserialize, Debug)]
struct SceneConfig {
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    frames_to_accumulate: u32,
    camera: CameraConfig,
    spheres: Vec<SphereConfig>,
    planes: Vec<PlaneConfig>,
    #[serde(default)]
    triangles: Vec<TriangleConfig>,
}

#[derive(Serialize, Deserialize, Debug)]
struct CameraConfig {
    fov_degrees: f32,
    char_aspect_ratio: f32,
}

#[derive(Serialize, Deserialize, Debug)]
struct SphereConfig {
    center: [f64; 3],
    radius: f64,
    color: [f64; 3],
    emission: [f64; 3],
    reflectivity: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct PlaneConfig {
    point: [f64; 3],
    normal: [f64; 3],
    color: [f64; 3],
    emission: [f64; 3],
    reflectivity: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct TriangleConfig {
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    color: [f64; 3],
    emission: [f64; 3],
    reflectivity: f64,
}

pub async fn run(full_color: bool, verbose: bool, scene_path: Option<&str>) {
    // Get terminal size to adjust output
    let (terminal_width, terminal_height) = terminal::size().unwrap();
    
    let scene_content = match scene_path {
        Some(path)=> std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read scene file at '{}': {}", path, e)),
        None => include_str!("scenes/Cornell_Box.json").to_string(),
    };

    let mut scene: SceneConfig = serde_json::from_str(&scene_content)
        .expect("Failed to parse embedded scene");

    // Ensure output fits in terminal
    scene.width = (terminal_width as u32).min(scene.width);
    scene.height = (terminal_height as u32 - 2).min(scene.height); // -2 for status lines

    // Initialize camera
    let mut camera = Camera::new(Vec3::new(0.0, 0.0, 0.0), -std::f32::consts::PI / 2.0, 0.0);

    let mut primitives: Vec<Primitive> = Vec::new();

    // Add spheres
    for s in &scene.spheres {
        primitives.push(Primitive::new_sphere(
            Vec3::new(s.center[0], s.center[1], s.center[2]),
            s.radius,
            Vec3::new(s.color[0], s.color[1], s.color[2]),
            Vec3::new(s.emission[0], s.emission[1], s.emission[2]),
            s.reflectivity,
        ));
    }

    // Add planes
    for p in &scene.planes {
        primitives.push(Primitive::new_plane(
            Vec3::new(p.point[0], p.point[1], p.point[2]),
            Vec3::new(p.normal[0], p.normal[1], p.normal[2]),
            Vec3::new(p.color[0], p.color[1], p.color[2]),
            Vec3::new(p.emission[0], p.emission[1], p.emission[2]),
            p.reflectivity,
        ));
    }

    // Add triangles
    for t in &scene.triangles {
        primitives.push(Primitive::new_triangle(
            Vec3::new(t.v0[0], t.v0[1], t.v0[2]),
            Vec3::new(t.v1[0], t.v1[1], t.v1[2]),
            Vec3::new(t.v2[0], t.v2[1], t.v2[2]),
            Vec3::new(t.color[0], t.color[1], t.color[2]),
            Vec3::new(t.emission[0], t.emission[1], t.emission[2]),
            t.reflectivity,
        ));
    }

    use wgpu::util::DeviceExt;

    fn primitive_aabb(p: &Primitive) -> (Vec3, Vec3) {
        match p.primitive_type {
            0 => { // sphere
                let r = p.sphere_radius;
                (p.sphere_center - Vec3::new_f32(r, r, r), p.sphere_center + Vec3::new_f32(r, r, r))
            }
            2 => { // triangle
                let min = Vec3::new_f32(
                    p.triangle_v0.x.min(p.triangle_v1.x).min(p.triangle_v2.x),
                    p.triangle_v0.y.min(p.triangle_v1.y).min(p.triangle_v2.y),
                    p.triangle_v0.z.min(p.triangle_v1.z).min(p.triangle_v2.z),
                );
                let max = Vec3::new_f32(
                    p.triangle_v0.x.max(p.triangle_v1.x).max(p.triangle_v2.x),
                    p.triangle_v0.y.max(p.triangle_v1.y).max(p.triangle_v2.y),
                    p.triangle_v0.z.max(p.triangle_v1.z).max(p.triangle_v2.z),
                );
                (min, max)
            }
            _ => (Vec3::new_f32(0.0,0.0,0.0), Vec3::new_f32(0.0,0.0,0.0)),
        }
    }

    fn build_uniform_grid(primitives: &Vec<Primitive>) -> (Vec3, Vec3, [u32;3], Vec<u32>, Vec<u32>) {
        // 1) scene bbox
        let mut bmin = Vec3::new_f32(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut bmax = Vec3::new_f32(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for p in primitives {
            let (mn, mx) = primitive_aabb(p);
            bmin = Vec3::new_f32(bmin.x.min(mn.x), bmin.y.min(mn.y), bmin.z.min(mn.z));
            bmax = Vec3::new_f32(bmax.x.max(mx.x), bmax.y.max(mx.y), bmax.z.max(mx.z));
        }
        // pad bbox slightly
        let pad = 1e-3f32;
        bmin = bmin - Vec3::new_f32(pad,pad,pad);
        bmax = bmax + Vec3::new_f32(pad,pad,pad);

        // 2) choose resolution (heuristic)
        let n = primitives.len() as f32;
        let extent = bmax - bmin;
        let s = ((n).powf(1.0/3.0)) * 1.5; // tweak factor
        let nx = (s * (extent.x / extent.x.max(extent.y).max(extent.z))).max(1.0).round() as u32;
        let ny = (s * (extent.y / extent.x.max(extent.y).max(extent.z))).max(1.0).round() as u32;
        let nz = (s * (extent.z / extent.x.max(extent.y).max(extent.z))).max(1.0).round() as u32;

        let nx = nx.max(1); let ny = ny.max(1); let nz = nz.max(1);

        // 3) bucket primitives
        let cell_count = (nx*ny*nz) as usize;
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); cell_count];

        let cell_size = Vec3::new_f32(
            (extent.x) / nx as f32,
            (extent.y) / ny as f32,
            (extent.z) / nz as f32,
        );

        for (idx, p) in primitives.iter().enumerate() {
            let (mn, mx) = primitive_aabb(p);
            let min_cell_x = ((mn.x - bmin.x) / cell_size.x).floor().clamp(0.0, (nx-1) as f32) as u32;
            let min_cell_y = ((mn.y - bmin.y) / cell_size.y).floor().clamp(0.0, (ny-1) as f32) as u32;
            let min_cell_z = ((mn.z - bmin.z) / cell_size.z).floor().clamp(0.0, (nz-1) as f32) as u32;
            let max_cell_x = ((mx.x - bmin.x) / cell_size.x).floor().clamp(0.0, (nx-1) as f32) as u32;
            let max_cell_y = ((mx.y - bmin.y) / cell_size.y).floor().clamp(0.0, (ny-1) as f32) as u32;
            let max_cell_z = ((mx.z - bmin.z) / cell_size.z).floor().clamp(0.0, (nz-1) as f32) as u32;

            for z in min_cell_z..=max_cell_z {
                for y in min_cell_y..=max_cell_y {
                    for x in min_cell_x..=max_cell_x {
                        let ci = (x + y * nx + z * nx * ny) as usize;
                        buckets[ci].push(idx as u32);
                    }
                }
            }
        }

        let mut offsets: Vec<u32> = Vec::with_capacity(cell_count + 1);
        let mut indices: Vec<u32> = Vec::new();
        offsets.push(0);
        for b in buckets.iter() {
            indices.extend(b.iter().cloned());
            offsets.push(indices.len() as u32);
        }

        let inv_cell_size = Vec3::new_f32(1.0 / cell_size.x, 1.0 / cell_size.y, 1.0 / cell_size.z);
        (bmin, inv_cell_size, [nx, ny, nz], offsets, indices)
    }

    let (grid_min, inv_cell_size, grid_dims, grid_offsets, grid_indices) = build_uniform_grid(&primitives);

    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.expect("Failed to find an appropriate adapter");
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.expect("Failed to create device");

    let grid_offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Grid Offsets"),
        contents: bytemuck::cast_slice(&grid_offsets),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let grid_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Grid Indices"),
        contents: bytemuck::cast_slice(&grid_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let limits = device.limits();
    if verbose {
        println!("Max workgroup size: {:?}", limits.max_compute_workgroup_size_x);
        println!("Max workgroup invocations: {}", limits.max_compute_invocations_per_workgroup);
    }

    let output_buffer_size = (scene.width * scene.height * std::mem::size_of::<Vec3>() as u32) as wgpu::BufferAddress;
    let accumulation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Accumulation Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Add variance buffer for the shader
    let variance_buffer_size = (scene.width * scene.height * 8) as wgpu::BufferAddress; // 8 bytes per VarianceBuffer (f32 + u32)
    let variance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Variance Buffer"),
        size: variance_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::bytes_of(&Uniforms {
            width: 0, height: 0, samples_per_pixel: 0, max_depth: 0,
            seed: 0, frame_number: 0, primitive_count: 0, _padding1: 0,
            aspect_ratio: 0.0, char_aspect_ratio: 0.0, fov_rad: 0.0, _padding2: 0.0,
            camera_pos: Vec3::new(0.0,0.0,0.0), camera_forward: Vec3::new(0.0,0.0,0.0),
            camera_right: Vec3::new(0.0,0.0,0.0), camera_up: Vec3::new(0.0,0.0,0.0),
            grid_min: grid_min,                       // use grid_min computed earlier
            inv_cell_size: inv_cell_size,             // use inv_cell_size computed earlier
            grid_dims: grid_dims,                     // use grid_dims computed earlier
            _padding3: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let primitive_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Primitive Buffer"),
        contents: bytemuck::cast_slice(&primitives),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"), 
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, 
        mapped_at_creation: false,
    });
    
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"), 
        size: output_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, 
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, 
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Enter raw mode and hide cursor
    terminal::enable_raw_mode().unwrap();
    execute!(io::stdout(), cursor::Hide).unwrap();
    
    let mut frame_count = 0;
    let mut camera_moved = true;
    
    // Frame rate tracking
    let mut last_frame_time = Instant::now();
    let mut frame_times = VecDeque::new();
    let max_frame_samples = 30; // Average over last 30 frames
    
    // Clear screen once at start
    print!("\x1B[2J\x1B[1;1H");
    io::stdout().flush().unwrap();
    
    // Create bind group once outside the loop
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"), 
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: primitive_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: accumulation_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: variance_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: grid_offsets_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: grid_indices_buffer.as_entire_binding() },
        ],
    });

    // Main loop
    loop {
        let total_frame_start = Instant::now();
        // Handle input
        let mut moved_this_frame = false;
        if event::poll(Duration::from_millis(1)).unwrap() {
            if let Event::Key(key) = event::read().unwrap() {
                let (forward, right, _) = camera.calculate_vectors();
                match key.code {
                    KeyCode::Char('w') => { camera.position = camera.position + forward * 0.1; moved_this_frame = true; }
                    KeyCode::Char('s') => { camera.position = camera.position - forward * 0.1; moved_this_frame = true; }
                    KeyCode::Char('a') => { camera.position = camera.position - right * 0.1; moved_this_frame = true; }
                    KeyCode::Char('d') => { camera.position = camera.position + right * 0.1; moved_this_frame = true; }
                    KeyCode::Up => { camera.pitch += 0.05; moved_this_frame = true; }
                    KeyCode::Down => { camera.pitch -= 0.05; moved_this_frame = true; }
                    KeyCode::Left => { camera.yaw -= 0.05; moved_this_frame = true; }
                    KeyCode::Right => { camera.yaw += 0.05; moved_this_frame = true; }
                    KeyCode::Esc => break, // Exit loop
                    _ => {}
                }
                camera.pitch = camera.pitch.clamp(-1.5, 1.5); // Clamp pitch
            }
        }

        if moved_this_frame {
            camera_moved = true;
            frame_count = 0; // Reset frame count for accumulation
        }
        
        // Render loop for temporal accumulation
        if frame_count < scene.frames_to_accumulate {
            // Update uniforms for this frame
            let (forward, right, up) = camera.calculate_vectors();
            let frame_uniforms = Uniforms {
                width: scene.width,
                height: scene.height,
                samples_per_pixel: scene.samples_per_pixel,
                max_depth: scene.max_depth,
                seed: rand::random::<u32>().wrapping_add(frame_count),
                frame_number: if camera_moved { 0 } else { frame_count },
                primitive_count: primitives.len() as u32, // ADDED
                _padding1: 0,
                aspect_ratio: scene.width as f32 / scene.height as f32,
                char_aspect_ratio: scene.camera.char_aspect_ratio,
                fov_rad: scene.camera.fov_degrees.to_radians(),
                _padding2: 0.0,
                camera_pos: camera.position,
                camera_forward: forward,
                camera_right: right,
                camera_up: up,
                grid_min: grid_min,               // ADDED — values computed earlier
                inv_cell_size: inv_cell_size,     // ADDED
                grid_dims: grid_dims,             // ADDED
                _padding3: 0,
            };
            
            // Update uniform buffer
            queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&frame_uniforms));
            
            // Render this frame (bind group is already created)
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(
                    (scene.width + 7) / 8,
                    (scene.height + 7) / 8, 
                    1);
            }
            queue.submit(Some(encoder.finish()));
            
            frame_count += 1;


            // Copy final result to staging buffer and render
            let mut display_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            display_encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
            queue.submit(Some(display_encoder.finish()));

            // Record gpu wait time to compare with cpu time to find bottlenecks
            let gpu_time = Instant::now();
            
            let buffer_slice = staging_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, |result| {
                result.unwrap();
            });
            let _ = device.poll(wgpu::MaintainBase::Wait);
            let gpu_done_time = Instant::now();
            let gpu_time_ms = (gpu_done_time.duration_since(gpu_time)).as_millis();
            
            let data = buffer_slice.get_mapped_range();
            let colors: &[Vec3] = bytemuck::cast_slice(&data);

            // Calculate frame rate
            let current_time = Instant::now();
            let frame_duration = current_time.duration_since(last_frame_time);
            last_frame_time = current_time;
            
            // Store frame time and maintain rolling average
            frame_times.push_back(frame_duration);
            if frame_times.len() > max_frame_samples {
                frame_times.pop_front();
            }
            
            // Calculate average FPS
            let avg_frame_time: Duration = frame_times.iter().sum::<Duration>() / frame_times.len() as u32;
            let fps = 1.0 / avg_frame_time.as_secs_f64();

            let cpu_start = Instant::now();

            // Move cursor to top-left
            print!("\x1B[1;1H");
            
            // Build the entire frame in memory in parallel (one String per row) to utilize rayon threads.
            use std::fmt::Write as FmtWrite;
            use rayon::prelude::*;

            let rows: Vec<String> = (0..scene.height).into_par_iter().map(|j| {
                let mut row = String::with_capacity((scene.width * 20) as usize);
                for i in 0..scene.width {
                    let index = (j * scene.width + i) as usize;
                    let pixel_color = colors[index];

                    if full_color {
                        let r = (pixel_color.x.sqrt() * 255.0).clamp(0.0, 255.0) as u8;
                        let g = (pixel_color.y.sqrt() * 255.0).clamp(0.0, 255.0) as u8;
                        let b = (pixel_color.z.sqrt() * 255.0).clamp(0.0, 255.0) as u8;
                        let _ = write!(row, "\x1b[38;2;{};{};{}m█\x1b[0m", r, g, b);
                    } else {
                        let gamma = 0.3f32;
                        let r = (pixel_color.x.powf(gamma) * 255.0).clamp(0.0, 255.0) as u8;
                        let g = (pixel_color.y.powf(gamma) * 255.0).clamp(0.0, 255.0) as u8;
                        let b = (pixel_color.z.powf(gamma) * 255.0).clamp(0.0, 255.0) as u8;

                        let brightness = 0.2126 * pixel_color.x + 0.7152 * pixel_color.y + 0.0722 * pixel_color.z;
                        let chars = [' ', '.', '\'', '`', '^', '"', ',', ':', ';', 'I', 'l', '!', 'i', '>', '<', '~', '+', '_', '-', '?', ']', '[', '}', '{', '1', ')', '(', '|', '\\', '/', 't', 'f', 'j', 'r', 'x', 'n', 'u', 'v', 'c', 'z', 'X', 'Y', 'U', 'J', 'C', 'L', 'Q', '0', 'O', 'Z', 'm', 'w', 'q', 'p', 'd', 'b', 'k', 'h', 'a', 'o', '*', '#', 'M', 'W', '&', '8', '%', 'B', '@', '$'];
                        let ci = (brightness.powf(gamma) * (chars.len() - 1) as f32)
                            .min((chars.len() - 1) as f32) as usize;
                        let _ = write!(row, "\x1b[38;2;{};{};{}m{}\x1b[0m", r, g, b, chars[ci]);
                    }
                }
                row.push_str("\r\n");
                row
            }).collect();

            // Combine rows into one mutable String so we can push the status line.
            let mut frame_buffer = rows.concat();

            let cpu_done_time = Instant::now();
            let cpu_time_ms = (cpu_done_time.duration_since(cpu_start)).as_millis();
            
            let io_start = Instant::now();
            print!("{}", frame_buffer);
            io::stdout().flush().unwrap();
            let io_time = io_start.elapsed().as_millis();

            let cleanup_start = Instant::now();
            drop(data);
            staging_buffer.unmap();
            let cleanup_time = cleanup_start.elapsed().as_millis();

            let total_frame_time = total_frame_start.elapsed().as_millis();
            let unaccounted = total_frame_time.saturating_sub(gpu_time_ms + cpu_time_ms + io_time + cleanup_time);

            // Add to status line:
            frame_buffer.push_str(&format!(
                "Frame: {}/{} | FPS: {:.1} | GPU: {}ms | CPU: {}ms | IO: {}ms | Other: {}ms | Total: {}ms\n",
                frame_count, scene.frames_to_accumulate, fps,
                gpu_time_ms, cpu_time_ms, io_time, unaccounted, total_frame_time
            ));
            
             // Output entire frame at once
             print!("{}", frame_buffer);
             io::stdout().flush().unwrap();
            
            camera_moved = false;
        } else {
            // If accumulation is finished, just wait for input
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    // Restore terminal state before exiting
    execute!(io::stdout(), cursor::Show).unwrap();
    terminal::disable_raw_mode().unwrap();
    println!("Exiting.");
}