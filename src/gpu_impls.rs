use cubecl::prelude::*;

#[cube(launch)]
pub fn vpin_gpu_kernel(
    buy_volumes: &Array<F32>,
    sell_volumes: &Array<F32>,
    output: &mut Array<F32>,
    window: UInt,
    len: UInt,
) {
    let idx = ABSOLUTE_POS;
    
    if idx >= len {
        return;
    }
    
    // Initialize output to 0.0 for positions before window
    if idx < window {
        output[idx] = F32::new(0.0);
        return;
    }
    
    // Calculate window sum for buy and sell volumes
    let mut buy_sum = F32::new(0.0);
    let mut sell_sum = F32::new(0.0);
    
    // Calculate the start index for the window
    let start_idx = idx + UInt::new(1) - window;
    
    // Sum over the window
    for i in range(start_idx, idx + UInt::new(1), Comptime::new(false)) {
        buy_sum += buy_volumes[i];
        sell_sum += sell_volumes[i];
    }
    
    let diff = buy_sum - sell_sum;
    let total = buy_sum + sell_sum;
    let imbalance = F32::abs(diff);
    
    if total > F32::new(1e-12) {
        output[idx] = imbalance / total;
    } else {
        output[idx] = F32::new(0.0);
    }
}

#[cfg(feature = "gpu")]
pub fn vpin_gpu_compute<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    buy_volumes: &[f64],
    sell_volumes: &[f64],
    window: usize,
) -> Vec<f64> {
    let len = buy_volumes.len().min(sell_volumes.len());
    
    // Convert f64 to f32 for GPU computation
    let buy_f32: Vec<f32> = buy_volumes.iter().map(|&x| x as f32).collect();
    let sell_f32: Vec<f32> = sell_volumes.iter().map(|&x| x as f32).collect();
    
    // Create GPU arrays
    let buy_gpu = client.create(f32::as_bytes(&buy_f32));
    let sell_gpu = client.create(f32::as_bytes(&sell_f32));
    let output_gpu = client.empty(len * core::mem::size_of::<f32>());
    
    // Launch kernel with proper grid/block configuration
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
    
    // Read back results
    let output_bytes = client.read(output_gpu.binding());
    let output_f32 = f32::from_bytes(&output_bytes);
    
    // Convert back to f64
    output_f32.iter().map(|&x| x as f64).collect()
}

#[cfg(feature = "cuda")]
pub fn vpin_cuda_compute(
    buy_volumes: &[f64],
    sell_volumes: &[f64],
    window: usize,
) -> Vec<f64> {
    use cubecl::cuda::{CudaDevice, CudaRuntime};
    use cubecl::Runtime;
    
    // Initialize CUDA device
    let device = CudaDevice::new(0);
    let client = CudaRuntime::client(&device);
    
    vpin_gpu_compute::<CudaRuntime>(&client, buy_volumes, sell_volumes, window)
}

#[cfg(not(feature = "cuda"))]
pub fn vpin_cuda_compute(
    buy_volumes: &[f64],
    sell_volumes: &[f64],
    window: usize,
) -> Vec<f64> {
    // Fallback to CPU when CUDA feature is not enabled
    crate::cpu_impls::vpin_cpu_kernel(buy_volumes, sell_volumes, window)
}

#[cfg(not(feature = "gpu"))]
pub fn vpin_gpu_compute<R>(
    _client: &R,
    buy_volumes: &[f64],
    sell_volumes: &[f64],
    window: usize,
) -> Vec<f64> {
    // Fallback to CPU when GPU feature is not enabled
    crate::cpu_impls::vpin_cpu_kernel(buy_volumes, sell_volumes, window)
}