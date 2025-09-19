use cubecl::prelude::*;
use cubecl_core as cubecl;

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

    if idx < window {
        output[idx] = F32::new(0.0);
        return;
    }

    let mut buy_sum = F32::new(0.0);
    let mut sell_sum = F32::new(0.0);

    let start_idx = idx + UInt::new(1) - window;

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

    if idx < UInt::new(2) {
        output[idx] = F32::new(0.0);
        return;
    }

    let current_diff = input[idx] - input[idx - UInt::new(1)];
    let coeff1 = F32::new(1.0) - alpha / F32::new(2.0);
    let coeff2 = F32::new(1.0) - alpha;
    let coeff3 = alpha - beta;

    output[idx] = coeff1 * current_diff + coeff2 * output[idx - UInt::new(1)]
        - coeff3 * output[idx - UInt::new(2)];
}

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
