# Terminal Raytracer

A Rust-based raytracer that renders 3D scenes directly in your terminal, using ASCII or full-color block characters. It leverages GPU compute (via wgpu) for fast, parallel rendering and supports temporal accumulation for denoising.

## Features

- Terminal output: ASCII or full-color rendering
- Adjustable samples per pixel and ray depth
- Temporal accumulation for smoother images
- Simple scene setup with spheres and a light source
- GPU-accelerated compute via wgpu

## Usage

Build and run with Cargo:

`cargo run --release`

For full-color output, use:

`cargo run --release -- --full-color`

## Dependencies

- Rust
- wgpu
- pollster
- bytemuck
