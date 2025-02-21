use rayon::prelude::*;
use rand::rng;
use rand_distr::{Distribution, Normal};
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, IntoPyArray};

#[pyfunction]
fn compute_gbm(
    py: Python<'_>,
    t_total: f64,
    n_steps: usize,
    mu: f64,
    sigma: f64,
    s0: f64,
    n_sims: usize,
    sample_start: usize,
    sample_stop: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let dt = t_total / n_steps as f64;
    let t: Vec<f64> = (0..=n_steps).map(|i| i as f64 * dt).collect();
    let drift = mu - 0.5 * sigma.powi(2);

    // Generate all paths in parallel using Rayon
    let paths: Vec<Vec<f64>> = (0..n_sims)
        .into_par_iter()
        .map(|_| {
            let mut rng = rng();
            let normal = Normal::new(0.0, 1.0).unwrap();
            
            // Generate Brownian increments
            let dw: Vec<f64> = (0..n_steps)
                .map(|_| normal.sample(&mut rng) * dt.sqrt())
                .collect();

            // Compute cumulative Brownian motion
            let mut w = vec![0.0];
            let mut current = 0.0;
            for &delta in &dw {
                current += delta;
                w.push(current);
            }

            // Compute GBM path
            t.iter()
                .zip(w.iter())
                .map(|(&ti, &wi)| s0 * (drift * ti + sigma * wi).exp())
                .collect()
        })
        .collect();

    // Extract samples from each path
    let samples: Vec<Vec<f64>> = paths
        .iter()
        .map(|path| path[sample_start..sample_stop].to_vec())
        .collect();

    // Convert to NumPy arrays
    let t_array = t.into_pyarray(py).into();  // Convert to Py<PyArray1>
    let samples_array = PyArray2::from_vec2(py, &samples).unwrap().into();  // Convert to Py<PyArray2>
    let paths_array = PyArray2::from_vec2(py, &paths).unwrap().into();  // Convert to Py<PyArray2>

    Ok((t_array, samples_array, paths_array))
}

#[pymodule]
fn gbm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_gbm, m)?);

    Ok(())
}