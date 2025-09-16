// vpin_kernel.rs - The actual GPU kernel implementation

#[cfg(feature = "gpu")]
use cubecl::prelude::*;
use bytemuck::cast_slice;

#[cfg(feature = "gpu")]
use pyo3::prelude::*;
#[cfg(feature = "gpu")]
use numpy::{PyArray1, PyReadonlyArray1};

#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn vpin_kernel<F: Float>(
    buy_volumes: &Array<F>,
    sell_volumes: &Array<F>,
    output: &mut Array<F>,
    window_size: UInt,
) {
    let idx = ABSOLUTE_POS;
    
    if idx < buy_volumes.len() {
        if idx < window_size {
            output[idx] = F::new(0.0);
        } else {
            // Rolling window VPIN calculation
            let mut buy_sum = F::new(0.0);
            let mut sell_sum = F::new(0.0);
            
            // Sum over window
            for i in range(0u32, window_size, Comptime::new(false)) {
                let window_idx = idx - window_size + i + 1;
                buy_sum += buy_volumes[window_idx];
                sell_sum += sell_volumes[window_idx];
            }
            
            let diff = buy_sum - sell_sum;
            let total = buy_sum + sell_sum;
            let imbalance = F::abs(diff);
            let epsilon = F::new(1e-12);
            
            if total > epsilon {
                output[idx] = imbalance / total;
            } else {
                output[idx] = F::new(0.0);
            }
        }
    }
}

// Standalone function for direct kernel usage
#[cfg(feature = "gpu")]
pub fn launch_vpin_kernel<R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    buy_volumes: &[f32],
    sell_volumes: &[f32], 
    window: u32,
) -> Vec<f32> {
    let len = buy_volumes.len().min(sell_volumes.len());
    
    // Create GPU arrays
    let buy_handle = client.create(cast_slice(buy_volumes));
    let sell_handle = client.create(cast_slice(sell_volumes));
    let output_handle = client.empty(len * core::mem::size_of::<f32>());
    
    // Calculate grid dimensions
    let cube_dim = CubeDim::new(256, 1, 1);
    let cube_count = CubeCount::Static(((len + 255) / 256) as u32, 1, 1);
    
    // Launch kernel
    unsafe {
        vpin_kernel::launch_unchecked::<F32, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(&buy_handle, len, 1),
            ArrayArg::from_raw_parts(&sell_handle, len, 1),
            ArrayArg::from_raw_parts(&output_handle, len, 1),
            ScalarArg::new(window),
        );
    }
    
    // Read results
    let bytes = client.read(output_handle.binding());
    cast_slice(&bytes).to_vec()
}

// Wrapper function for Python integration
#[cfg(feature = "gpu")]
pub fn vpin_kernel_wrapper<'py>(
    py: Python<'py>,
    device: &cubecl::wgpu::WgpuDevice,
    buy_volumes: PyReadonlyArray1<'py, f64>,
    sell_volumes: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    use cubecl::prelude::Runtime;
    use cubecl::wgpu::WgpuRuntime;
    
    let buy_vols = buy_volumes.as_array();
    let sell_vols = sell_volumes.as_array();
    let len = buy_vols.len().min(sell_vols.len());

    if len == 0 || window == 0 {
        return Ok(PyArray1::from_vec(py, vec![f64::NAN; len]).to_owned().into());
    }

    // Convert f64 to f32 for GPU computation
    let buy_vols_f32: Vec<f32> = buy_vols.iter().map(|&x| x as f32).collect();
    let sell_vols_f32: Vec<f32> = sell_vols.iter().map(|&x| x as f32).collect();

    // Get client from runtime
    let client = WgpuRuntime::client(device);

    // Launch kernel
    let results_f32 = launch_vpin_kernel::<WgpuRuntime>(
        &client,
        &buy_vols_f32,
        &sell_vols_f32,
        window as u32,
    );

    // Convert f32 back to f64
    let results: Vec<f64> = results_f32.iter().map(|&x| x as f64).collect();

    Ok(PyArray1::from_vec(py, results).to_owned().into())
}
