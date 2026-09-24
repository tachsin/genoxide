//! Batch fitness evaluation on the GPU: neuroevolution of a small neural network. Each genome is
//! the 65 weights of a network with 2 inputs, 16 hidden tanh units and 1 output. Its fitness is
//! the mean squared error of the network over 4,096 samples of a function: about 2 million
//! network evaluations per generation. A generation goes to the GPU at once, through `Batch`: one
//! upload of the weights, one dispatch with a workgroup per genome, one download of the errors.
//!
//! cargo run --release --manifest-path examples/gpu/Cargo.toml

use genoxide::prelude::*;
use std::sync::Mutex;
use std::time::Instant;
use wgpu::util::DeviceExt;

const HIDDEN: usize = 16;
// per hidden unit: a weight for each input and a bias; then the output weights and bias
const WEIGHTS: usize = HIDDEN * 3 + HIDDEN + 1;
const SAMPLES: usize = 4_096;
const POPULATION: usize = 512;
const GENERATIONS: u64 = 300;

// a workgroup per genome: its 256 invocations share the samples, then add up their errors
const SHADER: &str = "
const HIDDEN: u32 = 16u;
const WEIGHTS: u32 = 65u;

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> samples: array<vec4<f32>>; // x, y, target, unused
@group(0) @binding(2) var<storage, read_write> errors: array<f32>;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    let w = group.x * WEIGHTS;
    let count = arrayLength(&samples);
    var error = 0.0;
    for (var s = local; s < count; s = s + 256u) {
        let sample = samples[s];
        var output = weights[w + HIDDEN * 4u];
        for (var h = 0u; h < HIDDEN; h = h + 1u) {
            let unit = w + h * 3u;
            let activation = weights[unit] * sample.x + weights[unit + 1u] * sample.y + weights[unit + 2u];
            output = output + weights[w + HIDDEN * 3u + h] * tanh(activation);
        }
        let difference = output - sample.z;
        error = error + difference * difference;
    }
    partial[local] = error;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (local < stride) {
            partial[local] = partial[local] + partial[local + stride];
        }
        workgroupBarrier();
    }
    if (local == 0u) {
        errors[group.x] = partial[0] / f32(count);
    }
}
";

/// The samples: a grid over [-2, 2]², with the target sin(2x)·cos(y).
fn samples() -> Vec<[f64; 3]> {
    let side = (SAMPLES as f64).sqrt() as usize;
    let coordinate = |i: usize| -2.0 + 4.0 * i as f64 / (side - 1) as f64;
    (0..side * side)
        .map(|i| {
            let (x, y) = (coordinate(i % side), coordinate(i / side));
            [x, y, (2.0 * x).sin() * y.cos()]
        })
        .collect()
}

/// The network's mean squared error over the samples, on the CPU.
fn error(weights: &Reals, samples: &[[f64; 3]]) -> f64 {
    let total: f64 = samples
        .iter()
        .map(|&[x, y, target]| {
            let mut output = weights[HIDDEN * 4];
            for h in 0..HIDDEN {
                let unit = h * 3;
                let activation = weights[unit] * x + weights[unit + 1] * y + weights[unit + 2];
                output += weights[HIDDEN * 3 + h] * activation.tanh();
            }
            (output - target).powi(2)
        })
        .sum();
    total / samples.len() as f64
}

/// The network's error on the GPU, in single precision.
struct GpuError {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    samples: wgpu::Buffer,
    // the weights of a generation, to upload; reused
    staging: Mutex<Vec<f32>>,
}

impl GpuError {
    /// The GPU's device, with the samples uploaded, or `None` without a GPU.
    fn new(samples: &[[f64; 3]]) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        println!("GPU: {}", adapter.get_info().name);
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("error"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("error"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let packed: Vec<[f32; 4]> = samples
            .iter()
            .map(|&[x, y, target]| [x as f32, y as f32, target as f32, 0.0])
            .collect();
        let samples = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("samples"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Some(Self {
            device,
            queue,
            pipeline,
            samples,
            staging: Mutex::new(Vec::new()),
        })
    }

    /// The errors of the networks `genomes`, in their order.
    fn evaluate(&self, genomes: &[&Reals]) -> Vec<f64> {
        if genomes.is_empty() {
            return Vec::new();
        }
        let mut staging = self.staging.lock().unwrap();
        staging.clear();
        for genome in genomes {
            staging.extend(genome.iter().map(|&weight| weight as f32));
        }
        let weights = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("weights"),
                contents: bytemuck::cast_slice(&staging),
                usage: wgpu::BufferUsages::STORAGE,
            });
        drop(staging);
        let size = (genomes.len() * size_of::<f32>()) as u64;
        let errors = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("errors"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let download = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.samples.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: errors.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(genomes.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&errors, 0, &download, 0, size);
        self.queue.submit([encoder.finish()]);

        let slice = download.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| {
            result.expect("the GPU reads back the errors");
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the GPU finishes");
        let values: Vec<f64> = bytemuck::cast_slice::<u8, f32>(
            &slice.get_mapped_range().expect("the errors are mapped"),
        )
        .iter()
        .map(|&error| f64::from(error))
        .collect();
        download.unmap();
        values
    }
}

fn ga() -> genoxide::Result<Ga<Real, Tournament, SimulatedBinaryCrossover, PolynomialMutation>> {
    Ga::builder(Real::uniform(WEIGHTS, -3.0..=3.0)?)
        .population_size(POPULATION)
        .select(Tournament::new(3)?)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(2.0 / WEIGHTS as f64, 20.0)?)
        .minimize()
        .seed(1)
        .build()
}

fn main() -> genoxide::Result<()> {
    let samples = samples();
    let Some(gpu) = GpuError::new(&samples) else {
        println!("no GPU found: nothing to compare");
        return Ok(());
    };
    println!(
        "a {WEIGHTS}-weight network fitted to {SAMPLES} samples: {POPULATION} genomes, {GENERATIONS} generations\n"
    );

    let start = Instant::now();
    let outcome = Engine::new(ga()?, |weights: &Reals| error(weights, &samples))
        .parallel(true)
        .stop_when(Stop::generations(GENERATIONS))
        .run()?;
    println!(
        "CPU, every core: {:>6.2} s, error {:.4}",
        start.elapsed().as_secs_f64(),
        outcome.best_fitness()
    );

    let start = Instant::now();
    let outcome = Engine::new(ga()?, Batch(|genomes: &[&Reals]| gpu.evaluate(genomes)))
        .stop_when(Stop::generations(GENERATIONS))
        .run()?;
    println!(
        "GPU, batches:    {:>6.2} s, error {:.4} ({:.4} on the CPU in double precision)",
        start.elapsed().as_secs_f64(),
        outcome.best_fitness(),
        error(outcome.best_genome(), &samples)
    );
    Ok(())
}
