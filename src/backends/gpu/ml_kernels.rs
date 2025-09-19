use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch)]
pub fn matrix_vector_mul_add_kernel(
    matrix_a: &Array<F32>,
    vector_x: &Array<F32>,
    vector_b: &Array<F32>,
    output: &mut Array<F32>,
    rows: UInt,
    cols: UInt,
) {
    let row_idx = ABSOLUTE_POS;

    if row_idx >= rows {
        return;
    }

    let mut sum = F32::new(0.0);
    for col_idx in range(UInt::new(0), cols, Comptime::new(false)) {
        sum += matrix_a[row_idx * cols + col_idx] * vector_x[col_idx];
    }
    output[row_idx] = sum + vector_b[row_idx];
}

#[cube(launch)]
pub fn element_wise_tanh_kernel(input: &Array<F32>, output: &mut Array<F32>, len: UInt) {
    let idx = ABSOLUTE_POS;

    if idx >= len {
        return;
    }

    output[idx] = F32::tanh(input[idx]);
}

#[cube(launch)]
pub fn element_wise_sub_mul_kernel(
    input: &Array<F32>,
    output: &mut Array<F32>,
    sub_val: F32,
    mul_val: F32,
    len: UInt,
) {
    let idx = ABSOLUTE_POS;

    if idx >= len {
        return;
    }

    output[idx] = (input[idx] - sub_val) * mul_val;
}

#[cube(launch)]
pub fn reduction_sum_kernel(input: &Array<F32>, output: &mut Array<F32>, len: UInt) {
    let idx = ABSOLUTE_POS;

    if idx == UInt::new(0) {
        let mut sum = F32::new(0.0);
        for i in range(UInt::new(0), len, Comptime::new(false)) {
            sum += input[i];
        }
        output[UInt::new(0)] = sum;
    }
}

#[cube(launch)]
pub fn gradient_descent_update_kernel(
    weights: &mut Array<F32>,
    gradient: &Array<F32>,
    learning_rate: F32,
    len: UInt,
) {
    let idx = ABSOLUTE_POS;

    if idx >= len {
        return;
    }

    weights[idx] = weights[idx] - learning_rate * gradient[idx];
}

#[cube(launch)]
pub fn calculate_prediction_error_kernel(
    features: &Array<F32>,
    weights: &Array<F32>,
    labels: &Array<I32>,
    predictions: &mut Array<F32>,
    errors: &mut Array<F32>,
    n_features: UInt,
    n_samples: UInt,
) {
    let sample_idx = ABSOLUTE_POS;

    if sample_idx >= n_samples {
        return;
    }

    let mut weighted_sum = F32::new(0.0);
    for feature_idx in range(UInt::new(0), n_features, Comptime::new(false)) {
        weighted_sum += features[sample_idx * n_features + feature_idx] * weights[feature_idx];
    }

    let prediction = F32::tanh(weighted_sum);
    predictions[sample_idx] = prediction;

    let target = match labels[sample_idx] {
        I32::new(0) => F32::new(-1.0),
        I32::new(1) => F32::new(0.0),
        I32::new(2) => F32::new(1.0),
        _ => F32::new(0.0),
    };

    errors[sample_idx] = target - prediction;
}

#[cube(launch)]
pub fn accumulate_gradient_kernel(
    features: &Array<F32>,
    errors: &Array<F32>,
    sample_weights: &Array<F32>,
    gradient: &mut Array<F32>,
    n_features: UInt,
    n_samples: UInt,
) {
    let feature_idx = ABSOLUTE_POS;

    if feature_idx >= n_features {
        return;
    }

    let mut accumulated_gradient = F32::new(0.0);
    for sample_idx in range(UInt::new(0), n_samples, Comptime::new(false)) {
        let error = errors[sample_idx];
        let sample_weight = sample_weights[sample_idx];
        let feature_value = features[sample_idx * n_features + feature_idx];
        accumulated_gradient += error * feature_value * sample_weight;
    }
    gradient[feature_idx] = accumulated_gradient;
}

#[cube(launch)]
pub fn batch_prediction_kernel(
    features: &Array<F32>,
    weights: &Array<F32>,
    predictions: &mut Array<I32>,
    confidences: &mut Array<F32>,
    n_features: UInt,
    n_samples: UInt,
) {
    let sample_idx = ABSOLUTE_POS;

    if sample_idx >= n_samples {
        return;
    }

    let mut weighted_sum = F32::new(0.0);
    for feature_idx in range(UInt::new(0), n_features, Comptime::new(false)) {
        weighted_sum += features[sample_idx * n_features + feature_idx] * weights[feature_idx];
    }

    let normalized = F32::tanh(weighted_sum);
    confidences[sample_idx] = F32::abs(normalized).min(F32::new(1.0));

    predictions[sample_idx] = if normalized > F32::new(0.18) {
        I32::new(2)
    } else if normalized < F32::new(-0.18) {
        I32::new(0)
    } else {
        I32::new(1)
    };
}

#[cube(launch)]
pub fn evaluate_pattern_kernel(
    pattern_features: &Array<F32>,
    pattern_weights: &Array<F32>,
    output_scores: &mut Array<F32>,
    n_patterns: UInt,
    n_samples: UInt,
) {
    let sample_idx = ABSOLUTE_POS;

    if sample_idx >= n_samples {
        return;
    }

    let mut weighted_sum = F32::new(0.0);
    for pattern_idx in range(UInt::new(0), n_patterns, Comptime::new(false)) {
        weighted_sum +=
            pattern_features[sample_idx * n_patterns + pattern_idx] * pattern_weights[pattern_idx];
    }
    output_scores[sample_idx] = weighted_sum;
}

#[cube(launch)]
pub fn calculate_pattern_ensemble_weights_kernel(
    pattern_importance: &Array<F32>,
    output_weights: &mut Array<F32>,
    n_patterns: UInt,
) {
    let idx = ABSOLUTE_POS;

    if idx >= n_patterns {
        return;
    }

    output_weights[idx] = pattern_importance[idx];
}
