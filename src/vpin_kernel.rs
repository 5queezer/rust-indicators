// vpin_kernel.rs - Mathematically correct VPIN implementation

// CPU reference implementation for validation
pub fn vpin_cpu_reference(
    buy_volumes: &[f64],
    sell_volumes: &[f64],
    window: usize,
) -> Vec<f64> {
    let len = buy_volumes.len().min(sell_volumes.len());
    let mut output = vec![0.0; len];
    
    for i in window..len {
        let buy_sum: f64 = buy_volumes[i - window + 1..=i].iter().sum();
        let sell_sum: f64 = sell_volumes[i - window + 1..=i].iter().sum();
        
        let total = buy_sum + sell_sum;
        
        if total > 1e-12 {
            let diff = buy_sum - sell_sum;
            let imbalance = diff.abs();
            output[i] = imbalance / total;
        }
    }
    
    output
}