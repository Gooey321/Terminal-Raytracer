use terminal_raytracer::run;

use rayon::ThreadPoolBuilder;
use std::thread::available_parallelism;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    let full_color = args.iter().any(|arg| arg == "--full-color");
    let verbose = args.iter().any(|arg| arg == "--verbose");
    let scene_path = args.iter()
        .position(|arg| arg == "--path")
        .and_then(|i| args.get(i + 1));

    let thread_count = args.iter()
        .position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| available_parallelism().ok().map(|n| n.get() as usize))
        .unwrap_or(4);
    ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build_global()
        .expect("Failed to configure rayon thread pool");


    if full_color { 
        println!("outputting with █ characters"); 
    } else { 
        println!("outputting with ASCII characters"); 
    }

    if verbose {
        println!("rayon threads: {}", thread_count);
    }

    // Use pollster to run the async run function to completion.
    pollster::block_on(run(full_color, verbose, scene_path.map(String::as_str)));
}