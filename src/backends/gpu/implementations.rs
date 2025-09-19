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
pub fn vpin_cuda_compute(buy_volumes: &[f64], sell_volumes: &[f64], window: usize) -> Vec<f64> {
    use cubecl::cuda::{CudaDevice, CudaRuntime};
    use cubecl::Runtime;

    // Initialize CUDA device
    let device = CudaDevice::new(0);
    let client = CudaRuntime::client(&device);

    vpin_gpu_compute::<CudaRuntime>(&client, buy_volumes, sell_volumes, window)
}

#[cfg(not(feature = "cuda"))]
pub fn vpin_cuda_compute(buy_volumes: &[f64], sell_volumes: &[f64], window: usize) -> Vec<f64> {
    // Fallback to CPU when CUDA feature is not enabled
    crate::backends::cpu::implementations::vpin_cpu_kernel(buy_volumes, sell_volumes, window)
}

#[cfg(not(feature = "gpu"))]
pub fn vpin_gpu_compute<R>(
    _client: &R,
    buy_volumes: &[f64],
    sell_volumes: &[f64],
    window: usize,
) -> Vec<f64> {
    // Fallback to CPU when GPU feature is not enabled
    crate::backends::cpu::implementations::vpin_cpu_kernel(buy_volumes, sell_volumes, window)
}

/// GPU kernel for roofing high-pass filter
#[cube(launch)]
pub fn roofing_highpass_gpu_kernel(
    input: &Array<F32>,
    output: &mut Array<F32>,
    alpha: F32,
    beta: F32,
    len: UInt,
) {
    let idx = ABSOLUTE_POS;

    if idx >= len {
        return;
    }

    // Initialize first two values to 0
    if idx < UInt::new(2) {
        output[idx] = F32::new(0.0);
        return;
    }

    // Apply high-pass filter: (1 - alpha/2) * (data[i] - data[i-1]) + (1 - alpha) * out[i-1] - (alpha - beta) * out[i-2]
    let current_diff = input[idx] - input[idx - UInt::new(1)];
    let coeff1 = F32::new(1.0) - alpha / F32::new(2.0);
    let coeff2 = F32::new(1.0) - alpha;
    let coeff3 = alpha - beta;

    output[idx] = coeff1 * current_diff + coeff2 * output[idx - UInt::new(1)]
        - coeff3 * output[idx - UInt::new(2)];
}

/// GPU kernel for AGC normalization (requires sequential processing)
#[cube(launch)]
pub fn agc_normalize_gpu_kernel(
    input: &Array<F32>,
    output: &mut Array<F32>,
    peaks: &mut Array<F32>,
    decay_factor: F32,
    len: UInt,
) {
    let idx = ABSOLUTE_POS;

    if idx >= len {
        return;
    }

    // This kernel processes one element at a time in sequence
    // Note: This is not truly parallel due to dependencies, but can be used with proper synchronization
    if idx == UInt::new(0) {
        peaks[idx] = F32::abs(input[idx]);
    } else {
        peaks[idx] = decay_factor * peaks[idx - UInt::new(1)]
            + (F32::new(1.0) - decay_factor) * F32::abs(input[idx]);
    }

    if peaks[idx] > F32::new(1e-10) {
        output[idx] = input[idx] / peaks[idx];
    } else {
        output[idx] = F32::new(0.0);
    }
}

/// GPU kernel for quadrature calculation (one-bar difference)
#[cube(launch)]
pub fn quadrature_gpu_kernel(real_component: &Array<F32>, quadrature: &mut Array<F32>, len: UInt) {
    let idx = ABSOLUTE_POS;

    if idx >= len {
        return;
    }

    if idx == UInt::new(0) {
        quadrature[idx] = F32::new(0.0);
    } else {
        quadrature[idx] = real_component[idx] - real_component[idx - UInt::new(1)];
    }
}

#[cfg(feature = "gpu")]
pub fn hilbert_transform_gpu_compute<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    data: &[f64],
    lp_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    let len = data.len();

    // Early return for insufficient data
    if len < 50 {
        return (vec![0.0; len], vec![0.0; len]);
    }

    // Convert f64 to f32 for GPU computation
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    // Create GPU arrays
    let input_gpu = client.create(f32::as_bytes(&data_f32));
    let temp_gpu = client.empty(len * core::mem::size_of::<f32>());
    let _real_gpu = client.empty(len * core::mem::size_of::<f32>());
    let _imaginary_gpu = client.empty(len * core::mem::size_of::<f32>());
    let _peaks_gpu = client.empty(len * core::mem::size_of::<f32>());

    // Step 1: Apply roofing high-pass filter (48-period)
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

    // Step 2: Apply SuperSmoother low-pass filter (CPU fallback for sequential dependencies)
    let temp_bytes = client.read(temp_gpu.binding());
    let temp_f32 = f32::from_bytes(&temp_bytes);
    let temp_f64: Vec<f64> = temp_f32.iter().map(|&x| x as f64).collect();

    // Apply SuperSmoother on CPU (sequential dependencies)
    let roofed_data =
        crate::backends::cpu::implementations::apply_supersmoother(&temp_f64, lp_period as f64);

    // Step 3: Apply AGC normalization (CPU fallback due to sequential dependencies)
    let real_component =
        crate::backends::cpu::implementations::apply_agc_normalization_cpu(&roofed_data, 0.991);

    // Step 4: Calculate quadrature using GPU
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

    // Step 5: Apply AGC to quadrature and SuperSmoother (CPU fallback)
    let quadrature_bytes = client.read(quadrature_gpu.binding());
    let quadrature_f32 = f32::from_bytes(&quadrature_bytes);
    let quadrature_f64: Vec<f64> = quadrature_f32.iter().map(|&x| x as f64).collect();

    let agc_quadrature =
        crate::backends::cpu::implementations::apply_agc_normalization_cpu(&quadrature_f64, 0.991);

    // Apply final SuperSmoother to get imaginary component
    let imaginary_component =
        crate::backends::cpu::implementations::apply_supersmoother(&agc_quadrature, 10.0);

    (real_component, imaginary_component)
}

#[cfg(feature = "cuda")]
pub fn hilbert_transform_cuda_compute(data: &[f64], lp_period: usize) -> (Vec<f64>, Vec<f64>) {
    use cubecl::cuda::{CudaDevice, CudaRuntime};
    use cubecl::Runtime;

    // Initialize CUDA device
    let device = CudaDevice::new(0);
    let client = CudaRuntime::client(&device);

    hilbert_transform_gpu_compute::<CudaRuntime>(&client, data, lp_period)
}

#[cfg(not(feature = "cuda"))]
pub fn hilbert_transform_cuda_compute(data: &[f64], lp_period: usize) -> (Vec<f64>, Vec<f64>) {
    // Fallback to CPU when CUDA feature is not enabled
    crate::backends::cpu::implementations::hilbert_transform_core(data, lp_period)
}

#[cfg(not(feature = "gpu"))]
pub fn hilbert_transform_gpu_compute<R>(
    _client: &R,
    data: &[f64],
    lp_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    // Fallback to CPU when GPU feature is not enabled
    crate::backends::cpu::implementations::hilbert_transform_core(data, lp_period)
}
