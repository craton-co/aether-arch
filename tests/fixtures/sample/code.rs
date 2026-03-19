use std::collections::HashMap;

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn main() {
    let mut cache: HashMap<String, Vec<u64>> = HashMap::new();

    let sequence: Vec<u64> = (0..20).map(|n| fibonacci(n)).collect();
    println!("Fibonacci sequence: {:?}", sequence);

    cache.insert("fibonacci".to_string(), sequence.clone());
    cache.insert("squares".to_string(), (0..20).map(|n: u64| n * n).collect());
    cache.insert("cubes".to_string(), (0..20).map(|n: u64| n * n * n).collect());

    for (name, values) in &cache {
        let sum: u64 = values.iter().sum();
        let mean = sum as f64 / values.len() as f64;
        println!("{name}: sum={sum}, mean={mean:.2}, count={}", values.len());
    }
}
