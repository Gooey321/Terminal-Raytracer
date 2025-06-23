mod vec3;
mod sphere;

use std::time::Instant;
use vec3::Vec3;
use sphere::Sphere;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const MAX_DEPTH: u32 = 4;
const SAMPLES_PER_PIXEL: u32 = 256;
const FRAMES_TO_ACCUMULATE: u32 = 128;

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

async fn run(full_color: bool) {
    let width: u32 = 200;
    let height: u32 = 100;

    let spheres = vec![
        // Floor, Walls, Ceiling
        Sphere::new(Vec3::new(0.0, -1001.0, -3.0), 1000.0, Vec3::new(0.75, 0.75, 0.75), Vec3::new(0.0, 0.0, 0.0), 0.0),
        Sphere::new(Vec3::new(0.0, 1001.0, -3.0), 1000.0, Vec3::new(0.75, 0.75, 0.75), Vec3::new(0.0, 0.0, 0.0), 0.0),
        Sphere::new(Vec3::new(-1001.0, 0.0, -3.0), 1000.0, Vec3::new(0.75, 0.25, 0.25), Vec3::new(0.0, 0.0, 0.0), 0.0),
        Sphere::new(Vec3::new(1001.0, 0.0, -3.0), 1000.0, Vec3::new(0.25, 0.75, 0.25), Vec3::new(0.0, 0.0, 0.0), 0.0),
        Sphere::new(Vec3::new(0.0, 0.0, -1004.0), 1000.0, Vec3::new(0.75, 0.75, 0.75), Vec3::new(0.0, 0.0, 0.0), 0.0),
        // Central Spheres
        Sphere::new(Vec3::new(-0.5, -0.4, -2.5), 0.6, Vec3::new(0.9, 0.9, 0.9), Vec3::new(0.0, 0.0, 0.0), 0.8),
        Sphere::new(Vec3::new(0.5, -0.7, -3.2), 0.3, Vec3::new(0.6, 0.8, 0.9), Vec3::new(0.0, 0.0, 0.0), 0.2),
        // Light Source
        Sphere::new(Vec3::new(0.0, 0.8, -3.0), 0.3, Vec3::new(1.0, 1.0, 1.0), Vec3::new(5.0, 5.0, 5.0), 0.0),
    ];

    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.expect("Failed to find an appropriate adapter");
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.expect("Failed to create device");

    let uniforms = Uniforms {
        width, height,
        samples_per_pixel: SAMPLES_PER_PIXEL,
        max_depth: MAX_DEPTH,
        seed: rand::random(),
        aspect_ratio: width as f32 / height as f32,
        char_aspect_ratio: 0.55,
        fov_rad: 45.0f32.to_radians(),
        frame_number: 0,
        _padding1: 0,
        _padding2: 0,
        _padding3: 0.0,
    };
    
    let output_buffer_size = (width * height * std::mem::size_of::<Vec3>() as u32) as wgpu::BufferAddress;
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

    let output_buffer_size = (width * height * std::mem::size_of::<Vec3>() as u32) as wgpu::BufferAddress;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"), 
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"), size: output_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    // FIX: Create the layout and pipeline correctly
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
    for frame in 0..FRAMES_TO_ACCUMULATE {
        // Update uniforms for this frame
        let frame_uniforms = Uniforms {
            width, height,
            samples_per_pixel: SAMPLES_PER_PIXEL,
            max_depth: MAX_DEPTH,
            seed: rand::random::<u32>().wrapping_add(frame),
            aspect_ratio: width as f32 / height as f32,
            char_aspect_ratio: 0.55,
            fov_rad: 45.0f32.to_radians(),
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
            cpass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
        }
        queue.submit(Some(encoder.finish()));
        
        // Optional: show progress
        if frame % 16 == 0 || frame == FRAMES_TO_ACCUMULATE - 1 {
            println!("Frame {}/{}", frame + 1, FRAMES_TO_ACCUMULATE);
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

    for j in 0..height {
        for i in 0..width {
            let index = (j * width + i) as usize;
            write_color(colors[index], full_color);
        }
        println!();
    }
    drop(data);
    staging_buffer.unmap();

    let duration = start.elapsed();
    println!("Temporal accumulation completed in: {:?}", duration);
    println!("Total effective samples: {}", SAMPLES_PER_PIXEL * FRAMES_TO_ACCUMULATE);
    println!("Image dimensions: {}x{}", width, height);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let full_color = args.contains(&"--full-color".to_string());
    
    if full_color { println!("outputting with █ characters"); } 
    else { println!("outputting with ASCII characters"); }

    // Use pollster to run the async run function to completion.
    pollster::block_on(run(full_color));
}