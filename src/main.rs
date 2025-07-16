mod vec3;
mod sphere;

use std::time::Instant;
use vec3::Vec3;
use sphere::Sphere;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use serde::{Serialize, Deserialize};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    seed: u32,
    frame_number: u32,
    _padding1: u32,
    _padding2: u32,
    aspect_ratio: f32,
    char_aspect_ratio: f32,
    fov_rad: f32,
    _padding3: f32,
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

fn load_scene(filename: &str) -> Result<SceneConfig, Box<dyn std::error::Error>> {
    let file_content = std::fs::read_to_string(filename)?;
    let scene: SceneConfig = serde_json::from_str(&file_content)?;
    Ok(scene)
}

fn write_color(pixel_color: Vec3, full_color: bool) {
    let r = (pixel_color.x.sqrt() * 255.0).clamp(0.0, 255.0) as u8;
    let g = (pixel_color.y.sqrt() * 255.0).clamp(0.0, 255.0) as u8;
    let b = (pixel_color.z.sqrt() * 255.0).clamp(0.0, 255.0) as u8;

    let brightness = 0.2126 * pixel_color.x + 0.7152 * pixel_color.y + 0.0722 * pixel_color.z;

    if full_color {
        print!("\x1b[38;2;{};{};{}m█\x1b[0m", r, g, b);
    } else {
        let chars = ['@', '%', '#', 'a', '*', '+', '=', '-', ':', '.', '@'];
        let index = (brightness.sqrt() * (chars.len() - 1) as f32).min((chars.len() - 1) as f32) as usize;
        print!("\x1b[38;2;{};{};{}m{}\x1b[0m", r, g, b, chars[index]);
    }
}

async fn run(full_color: bool, verbose: bool,) {
    let scene = load_scene("src/scene.json").expect("Failed to load scene");
    
    let spheres: Vec<Sphere> = scene.spheres.iter().map(|s| {
        Sphere::new(
            Vec3::new(s.center[0], s.center[1], s.center[2]),
            s.radius,
            Vec3::new(s.color[0], s.color[1], s.color[2]),
            Vec3::new(s.emission[0], s.emission[1], s.emission[2]),
            s.reflectivity,
        )
    }).collect();

    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.expect("Failed to find an appropriate adapter");
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.expect("Failed to create device");

    let limits = device.limits();
    println!("Max workgroup size: {:?}", limits.max_compute_workgroup_size_x);
    println!("Max workgroup invocations: {}", limits.max_compute_invocations_per_workgroup);

    let uniforms = Uniforms {
        width: scene.width,
        height: scene.height,
        samples_per_pixel: scene.samples_per_pixel,
        max_depth: scene.max_depth,
        seed: rand::random(),
        aspect_ratio: scene.width as f32 / scene.height as f32,
        char_aspect_ratio: scene.camera.char_aspect_ratio,
        fov_rad: scene.camera.fov_degrees.to_radians(),
        frame_number: 0,
        _padding1: 0,
        _padding2: 0,
        _padding3: 0.0,
    };

    let output_buffer_size = (scene.width * scene.height * std::mem::size_of::<Vec3>() as u32) as wgpu::BufferAddress;
    let accumulation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Accumulation Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let sphere_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sphere Buffer"),
        contents: bytemuck::cast_slice(&spheres),
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

    println!("Starting temporal accumulation render...");
    let start = Instant::now();
    
    // Render loop for temporal accumulation
    for frame in 0..scene.frames_to_accumulate {
        // Update uniforms for this frame
        let frame_uniforms = Uniforms {
            width: scene.width,
            height: scene.height,
            samples_per_pixel: scene.samples_per_pixel,
            max_depth: scene.max_depth,
            seed: rand::random::<u32>().wrapping_add(frame),
            aspect_ratio: scene.width as f32 / scene.height as f32,
            char_aspect_ratio: scene.camera.char_aspect_ratio,
            fov_rad: scene.camera.fov_degrees.to_radians(),
            frame_number: frame,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0.0,
        };
        
        // Update uniform buffer
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&frame_uniforms));
        
        // Create bind group for this frame
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"), 
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sphere_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: accumulation_buffer.as_entire_binding() },
            ],
        });

        // Render this frame
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                (scene.width + 15) / 16,
                (scene.height + 15) / 16, 
                1);
        }
        queue.submit(Some(encoder.finish()));
        
        // Show progress if verbose mode is enabled
        if verbose && (frame % 16 == 0 || frame == scene.frames_to_accumulate - 1) {
            println!("Frame {}/{}", frame + 1, scene.frames_to_accumulate);
        }
    }

    // Copy final result to staging buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let mapping_complete = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mapping_complete_clone = mapping_complete.clone();
    
    buffer_slice.map_async(wgpu::MapMode::Read, move |_result| {
        mapping_complete_clone.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    
    // Poll the device until mapping is complete
    while !mapping_complete.load(std::sync::atomic::Ordering::Relaxed) {
        let _ = device.poll(wgpu::MaintainBase::Poll);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    let data = buffer_slice.get_mapped_range();
    let colors: &[Vec3] = bytemuck::cast_slice(&data);

    for j in 0..scene.height {
        for i in 0..scene.width {
            let index = (j * scene.width + i) as usize;
            write_color(colors[index], full_color);
        }
        println!();
    }
    drop(data);
    staging_buffer.unmap();

    let duration = start.elapsed();
    println!("Temporal accumulation completed in: {:?}", duration);
    let time_per_pixel = duration.as_secs_f64() / (scene.width * scene.height) as f64;
    println!("Time per pixel: {:.6} seconds", time_per_pixel);
    println!("Total effective samples: {}", scene.samples_per_pixel * scene.frames_to_accumulate);
    println!("Image dimensions: {}x{}", scene.width, scene.height);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let full_color = args.contains(&"--full-color".to_string());
    let verbose = args.contains(&"--verbose".to_string());
    
    if full_color { println!("outputting with █ characters"); } 
    else { println!("outputting with ASCII characters"); }

    if verbose { println!("Verbose mode enabled"); }
    else { println!("Detailed output disabled"); }

    // Use pollster to run the async run function to completion.
    pollster::block_on(run(full_color, verbose));
}