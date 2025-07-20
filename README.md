# Terminal Raytracer

A raytracer that renders 3D scenes directly in the terminal using ASCII characters or full-color Unicode blocks "█". Built with Rust and wgpu.

Made for HackClubs Summer Of Hacking

## Features

### Rendering

- **Real-time GPU raytracing** using compute shaders (wgpu)
- **Multiple output modes**: ASCII art or full-color block characters
- **Physically-based rendering** with proper lighting and materials
- **Temporal accumulation** for progressive image refinement
- **Adaptive sampling** for improved quality in complex areas
- **Interactive camera controls** with WASD movement and arrow key rotation

## How to run it yourself

### Prerequisites

- Rust (latest stable)
- GPU with WGPU suppot
- A really fast terminal (I used the Kitty Terminal, iTerm2 isn't fast enough)

#### Note: Only tested on a Macbook Pro M3

### Build and Run

```bash
# Clone and build
cargo build --release

# Run with ASCII output
cargo run --release

# Run with full-color blocks
cargo run --release -- --full-color

# Enable verbose output
cargo run --release -- --verbose
```

### Controls

- **WASD**: Move camera (forward/back/left/right)
- **Arrow keys**: Look around (pitch/yaw)
- **ESC**: Exit

## Scene Configuration

Scenes are defined in JSON in [`src/scenes/`](src/scenes/). Each scene can contain:

### Basic Settings

```json
{
  "width": 400,
  "height": 200,
  "samples_per_pixel": 16,
  "max_depth": 32,
  "frames_to_accumulate": 100000000,
  "camera": {
    "fov_degrees": 50.0,
    "char_aspect_ratio": 0.55
  }
}
```

### Objects

```json
{
  "spheres": [
    {
      "center": [0.0, 0.0, -3.0],
      "radius": 0.5,
      "color": [0.8, 0.2, 0.2],
      "emission": [0.0, 0.0, 0.0],
      "reflectivity": 0.3
    }
  ],
  "planes": [
    {
      "point": [0.0, -1.0, 0.0],
      "normal": [0.0, 1.0, 0.0],
      "color": [0.6, 0.6, 0.6],
      "emission": [0.0, 0.0, 0.0],
      "reflectivity": 0.1
    }
  ],
  "triangles": [
    {
      "v0": [-1.0, 1.0, -4.0],
      "v1": [1.0, 1.0, -4.0],
      "v2": [0.0, 2.0, -4.0],
      "color": [0.2, 0.8, 0.2],
      "emission": [0.0, 0.0, 0.0],
      "reflectivity": 0.5
    }
  ]
}
```

### Rendering Techniques

- **Path Tracing**: Physically accurate light transport simulation
- **Next Event Estimation**: Direct light sampling for faster convergence
- **Russian Roulette**: Probabilistic ray termination for efficiency
- **Cosine-weighted Sampling**: Importance sampling for Lambertian surfaces

### Performance Optimizations

- **Adaptive Sampling**: More samples in high-variance areas
- **Temporal Accumulation**: Progressive refinement over multiple frames
- **Efficient Display**: Only renders every 5th frame for interactivity
- **GPU Memory Management**: Optimized buffer usage and staging
