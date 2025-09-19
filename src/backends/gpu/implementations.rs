use crate::backends::gpu::indicator_kernels::*;
use crate::backends::gpu::ml_kernels::*;
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cfg(feature = "gpu")]
pub fn vpin_gpu_compute<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    buy_volumes: &[f64],
    sell_volumes: &[f64],
    window: usize,
) -> Vec<f64> {
    let len = buy_volumes.len().min(sell_volumes.len());

    let buy_f32: Vec<f32> = buy_volumes.iter().map(|&x| x as f32).collect();
    let sell_f32: Vec<f32> = sell_volumes.iter().map(|&x| x as f32).collect();

    let buy_gpu = client.create(f32::as_bytes(&buy_f32));
    let sell_gpu = client.create(f32::as_bytes(&sell_f32));
    let output_gpu = client.empty(len * core::mem::size_of::<f32>());

    let threads_per_block = 256u32;
    let blocks = (len as u32 + threads_per_block - 1) / threads_per_block;

    unsafe {
        vpin_gpu_kernel::launch::<R>(
            client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new(threads_per_block, 1, 1),
            ArrayArg::from_raw_parts(&buy_gpu, len, 1),
            ArrayArg::from_raw_parts(&sell_gpu, len, 1),
            ArrayArg::from_raw_parts(&output_gpu, len, 1),
            ScalarArg::new(window as u32),
            ScalarArg::new(len as u32),
        );
    }

    let output_bytes = client.read(output_gpu.binding());
    let output_f32 = f32::from_bytes(&output_bytes);

    output_f32.iter().map(|&x| x as f64).collect()
}

#[cfg(feature = "cuda")]
pub fn vpin_cuda_compute(buy_volumes: &[f64], sell_volumes: &[f64], window: usize) -> Vec<f64> {
    use cubecl::cuda::{CudaDevice, CudaRuntime};
    use cubecl::Runtime;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::client(&device);

    vpin_gpu_compute::<CudaRuntime>(&client, buy_volumes, sell_volumes, window)
}

#[cfg(not(feature = "cuda"))]
pub fn vpin_cuda_compute(buy_volumes: &[f64], sell_volumes: &[f64], window: usize) -> Vec<f64> {
    crate::backends::cpu::implementations::vpin_cpu_kernel(buy_volumes, sell_volumes, window)
}

#[cfg(not(feature = "gpu"))]
pub fn vpin_gpu_compute<R>(
    _client: &R,
    buy_volumes: &[f64],
    sell_volumes: &[f64],
    window: usize,
) -> Vec<f64> {
    crate::backends::cpu::implementations::vpin_cpu_kernel(buy_volumes, sell_volumes, window)
}

#[cfg(feature = "gpu")]
pub fn hilbert_transform_gpu_compute<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    data: &[f64],
    lp_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let len = data.len();

    if len < 50 {
        return (vec![0.0; len], vec![0.0; len]);
    }

    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let input_gpu = client.create(f32::as_bytes(&data_f32));
    let temp_gpu = client.empty(len * core::mem::size_of::<f32>());

    let hp_period = 48.0f32;
    let alpha = (2.0 * std::f32::consts::PI / hp_period).cos();
    let beta = (2.0 * std::f32::consts::PI / hp_period).sin();
    let gamma = 1.0 / (alpha + beta);
    let alpha_adj = gamma * alpha;
    let beta_adj = gamma * beta;

    let threads_per_block = 256u32;
    let blocks = (len as u32 + threads_per_block - 1) / threads_per_block;

    unsafe {
        roofing_highpass_gpu_kernel::launch::<R>(
            client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new(threads_per_block, 1, 1),
            ArrayArg::from_raw_parts(&input_gpu, len, 1),
            ArrayArg::from_raw_parts(&temp_gpu, len, 1),
            ScalarArg::new(alpha_adj),
            ScalarArg::new(beta_adj),
            ScalarArg::new(len as u32),
        );
    }

    let temp_bytes = client.read(temp_gpu.binding());
    let temp_f32 = f32::from_bytes(&temp_bytes);
    let temp_f64: Vec<f64> = temp_f32.iter().map(|&x| x as f64).collect();

    let roofed_data =
        crate::backends::cpu::implementations::apply_supersmoother(&temp_f64, lp_period as f64);

    let real_component =
        crate::backends::cpu::implementations::apply_agc_normalization_cpu(&roofed_data, 0.991);

    let real_f32: Vec<f32> = real_component.iter().map(|&x| x as f32).collect();
    let real_gpu_final = client.create(f32::as_bytes(&real_f32));
    let quadrature_gpu = client.empty(len * core::mem::size_of::<f32>());

    unsafe {
        quadrature_gpu_kernel::launch::<R>(
            client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new(threads_per_block, 1, 1),
            ArrayArg::from_raw_parts(&real_gpu_final, len, 1),
            ArrayArg::from_raw_parts(&quadrature_gpu, len, 1),
            ScalarArg::new(len as u32),
        );
    }

    let quadrature_bytes = client.read(quadrature_gpu.binding());
    let quadrature_f32 = f32::from_bytes(&quadrature_bytes);
    let quadrature_f64: Vec<f64> = quadrature_f32.iter().map(|&x| x as f64).collect();

    let agc_quadrature =
        crate::backends::cpu::implementations::apply_agc_normalization_cpu(&quadrature_f64, 0.991);

    let imaginary_component =
        crate::backends::cpu::implementations::apply_supersmoother(&agc_quadrature, 10.0);

    (real_component, imaginary_component)
}

#[cfg(feature = "cuda")]
pub fn hilbert_transform_cuda_compute(data: &[f64], lp_period: usize) -> (Vec<f64>, Vec<f64>) {
    use cubecl::cuda::{CudaDevice, CudaRuntime};
    use cubecl::Runtime;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::client(&device);

    hilbert_transform_gpu_compute::<CudaRuntime>(&client, data, lp_period)
}

#[cfg(not(feature = "cuda"))]
pub fn hilbert_transform_cuda_compute(data: &[f64], lp_period: usize) -> (Vec<f64>, Vec<f64>) {
    crate::backends::cpu::implementations::hilbert_transform_core(data, lp_period)
}

#[cfg(not(feature = "gpu"))]
pub fn hilbert_transform_gpu_compute<R>(
    _client: &R,
    data: &[f64],
    lp_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    crate::backends::cpu::implementations::hilbert_transform_core(data, lp_period)
}
