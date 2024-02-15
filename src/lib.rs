use itertools::Itertools;
use ndarray::{s, Array, Array1};
use ndarray_rand::{
    rand::{distributions::Uniform, SeedableRng},
    RandomExt,
};
use numpy::{PyArray1, ToPyArray};
use pad::PadStr;
use pyo3::{
    prelude::*,
    types::{PyDict, PyString},
};
use rand_isaac::isaac64::Isaac64Rng;
use rayon::prelude::*;
use std::collections::HashMap;

fn generate_hdv<'py>(length: usize, rng_obj: &mut Isaac64Rng) -> Array1<i8> {
    Array::random_using(length, Uniform::from(0_i8..2_i8), rng_obj) * 2 - 1
}

fn bind(arrays: Vec<Array1<i8>>) -> Array1<i8> {
    // np.multiply.reduce(np.stack(hdv_list))
    let mut out: Array1<i8> = Array1::ones(arrays[0].len());
    for array in arrays {
        out = out * array;
    }
    out
}

fn shift(arr: &Array1<i8>, by: isize) -> Array1<i8> {
    // np.roll(arr, by)
    let x = by % arr.len() as isize;
    let mut new_array = Array1::ones(arr.len());
    arr.slice(s![-x..]).assign_to(new_array.slice_mut(s![..x]));
    arr.slice(s![x..]).assign_to(new_array.slice_mut(s![..-x]));
    new_array
}

fn generate_trimer_hdvs(amino_acids: &str, seed: u64) -> HashMap<String, Array1<i8>> {
    // Generate HDVs for each amino acid
    let mut rng_obj = Isaac64Rng::seed_from_u64(seed);
    let aa_hdvs: HashMap<char, Array1<i8>> = amino_acids
        .chars()
        .map(|aa| (aa, generate_hdv(HDV_LENGTH, &mut rng_obj)))
        .collect();
    // Use these to generate composite HDVs for all trimers
    itertools::iproduct!(
        amino_acids.chars(),
        amino_acids.chars(),
        amino_acids.chars()
    )
    .map(|(aa1, aa2, aa3)| {
        let trimer: String = format!("{}{}{}", &aa1, &aa2, &aa3);
        let this_hdv = bind(vec![
            shift(&aa_hdvs[&aa1], 2),
            shift(&aa_hdvs[&aa2], 1),
            aa_hdvs[&aa3].clone(),
        ]);
        (trimer, this_hdv)
    })
    .collect()
}

#[pyfunction]
fn py_generate_trimer_hdvs<'py>(
    py: Python<'py>,
    amino_acids: &str,
    seed: u64,
) -> PyResult<&'py PyDict> {
    // Run internal trimer function
    let trimers_hashmap = generate_trimer_hdvs(amino_acids, seed);
    // Convert hashmap to PyDict and return
    let trimer_hdvs = PyDict::new(py);
    for (trimer, hdv) in trimers_hashmap {
        trimer_hdvs.set_item(trimer, hdv.to_pyarray(py))?;
    }
    Ok(trimer_hdvs)
}

#[pyfunction]
fn embed_sequences<'py>(
    py: Python<'py>,
    sequences: Vec<String>,
    trimers: &PyDict,
) -> Vec<&'py PyArray1<i8>> {
    // Get trimer hashmap
    let trimers_hashmap: HashMap<String, Array1<i8>> = trimers
        .iter()
        .map(|(key, value)| {
            let conv_key = key.downcast::<PyString>().unwrap().to_string();
            unsafe {
                let conv_value = value
                    .downcast::<PyArray1<i8>>()
                    .unwrap()
                    .as_array()
                    .to_owned();
                (conv_key, conv_value)
            }
        })
        .collect();
    // Iterate over sequences and embed them
    let arrays: Vec<Array1<i8>> = sequences
        .par_iter()
        .map(|seq| {
            // Pad sequence to allow scanning of last base
            let pad_seq = &seq.pad_to_width_with_char(seq.len() + 2, '.');
            // Add together all the hdvs
            let mut seq_hdv = Array1::<i8>::zeros(HDV_LENGTH);
            for (aa1, aa2, aa3) in pad_seq.chars().tuple_windows() {
                let trimer: String = format!("{}{}{}", &aa1, &aa2, &aa3);
                seq_hdv = seq_hdv + trimers_hashmap[&trimer].clone();
            }
            // Convert to sign
            seq_hdv.map(|x| if x.signum() != 0 { x.signum() } else { 1 })
        })
        .collect();
    arrays.iter().map(|a| a.to_pyarray(py)).collect()
}

#[pyfunction]
fn embed_sequences_positional<'py>(
    py: Python<'py>,
    sequences: Vec<String>,
    trimer_hdvs: &PyDict,
    position_hdvs: Vec<&'py PyArray1<i8>>,
) -> Vec<&'py PyArray1<i8>> {
    // Get trimer hashmap
    let trimers_hashmap: HashMap<String, Array1<i8>> = trimer_hdvs
        .iter()
        .map(|(key, value)| {
            let conv_key = key.downcast::<PyString>().unwrap().to_string();
            unsafe {
                let conv_value = value
                    .downcast::<PyArray1<i8>>()
                    .unwrap()
                    .as_array()
                    .to_owned();
                (conv_key, conv_value)
            }
        })
        .collect();
    // Convert position hdvs into rust object
    let position_hdv_vec: Vec<Array1<i8>> = position_hdvs
        .iter()
        .map(|x| unsafe { x.as_array().to_owned() })
        .collect();
    // Iterate over sequences and embed them
    let arrays: Vec<Array1<i8>> = sequences
        .par_iter()
        .map(|seq| {
            // Pad sequence to allow scanning of last base
            let pad_seq = &seq.pad_to_width_with_char(seq.len() + 2, '.');
            let seq_len = pad_seq.chars().count();
            // Add together all the hdvs
            let mut seq_hdv = Array1::<i8>::zeros(HDV_LENGTH);
            for (ind, (aa1, aa2, aa3)) in pad_seq.chars().tuple_windows().enumerate() {
                let pos_ind = (ind / seq_len) * position_hdv_vec.len();
                let trimer: String = format!("{}{}{}", &aa1, &aa2, &aa3);
                seq_hdv = seq_hdv
                    + bind(vec![
                        position_hdv_vec[pos_ind].clone(),
                        trimers_hashmap[&trimer].clone(),
                    ]);
            }
            // Convert to sign
            seq_hdv.map(|x| if x.signum() != 0 { x.signum() } else { 1 })
        })
        .collect();
    arrays.iter().map(|a| a.to_pyarray(py)).collect()
}

const HDV_LENGTH: usize = 10000;

/// A Python module implemented in Rust.
#[pymodule]
fn hyperdim_proteins(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_generate_trimer_hdvs, m)?)?;
    m.add_function(wrap_pyfunction!(embed_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(embed_sequences_positional, m)?)?;
    Ok(())
}
