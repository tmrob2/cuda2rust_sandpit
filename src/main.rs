use rand::Rng;
use ctest::{doubler_interface, cuda_spaxy_ffi, CS_SI, SP_TYPE};

fn main() {
    println!("Hello, world!");

    let mut rng = rand::thread_rng();

    let x: Vec<i32> = (1..10).map(|_| rng.gen_range(0..10)).collect();
    let y: Vec<i32> = (1..10).map(|_| rng.gen_range(0..10)).collect();
    
    println!("Testing C interface; creating a vector of doubled values.");

    let output: Vec<i32> = x.iter().map(|z| doubler_interface(*z)).collect();

    println!("C FFI output => {:?}", output);

    println!("Testing C interface; testing add function with ptrs.");

    println!("Testing startup and teardown of GPU");


    let x: Vec<f32> = (1..10).map(|_| rng.gen_range(0.0..10.0)).collect();
    let y: Vec<f32> = (1..10).map(|_| rng.gen_range(0.0..10.0)).collect();

    //cuda_test(&x[..], &y[..]);

    let mut y: Vec<f32> = (1..10).map(|_| rng.gen_range(0.0..10.0)).collect();
    let alpha = 1.0;
    cuda_spaxy_ffi(&x[..], &mut y[..], alpha);

    println!("Testing custom sparse matrix structure");

    let mut csA: CS_SI = CS_SI { 
        nzmax: 1, 
        m: 2, 
        n: 2, 
        p: vec![1], 
        i: vec![1], 
        x: vec![1.0], 
        nz: 1,
        sptype: SP_TYPE::Triple
    };

    println!("Testing COO to CSR");

    let row: Vec<i32> = vec![0, 1, 2, 3];
    let col: Vec<i32> = vec![0, 1, 2, 1];
    let vals: Vec<f32> = vec![5., 8., 3., 6.];
    let nzmax = 4; 
    let nz = 0;
    let m = 4;
    let n = 4;
    let mut coo = CS_SI::make(nzmax, m, n, nz, SP_TYPE::Triple);
    for k in 0..vals.len() {
        coo.triple_entry(row[k], col[k], vals[k]);
    }
    let csr = coo.csr_compress();

    println!("csr\n{:?}", csr);

}
