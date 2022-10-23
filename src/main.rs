use libc::c_float;
use rand::Rng;
use ctest::*;
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

    println!("Testing Sp CSR Mv");

    let row: Vec<i32> = vec![0, 0, 0, 1, 2, 2, 2, 3, 3];
    let col: Vec<i32> = vec![0, 2, 3, 1, 0, 2, 3, 1, 3]; 
    let val: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let nzmax = val.len() as i32;
    let nz = 0;
    let m = 4;
    let n = 4;
    let mut coo = CS_SI::make(nzmax, m, n, nz, SP_TYPE::Triple);
    for k in 0..val.len() {
        coo.triple_entry(row[k], col[k], val[k]);
    }
    let csr = coo.csr_compress();

    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mut y: Vec<f32> = vec![0., 0., 0., 0.];

    csr_spmv_ffi(
        &csr.i, 
        &csr.p, 
        &csr.x, 
        &x, 
        &mut y,
        csr.nz, 
        csr.i.len() as i32, 
        csr.m, 
        csr.n
    );

    println!("y: {:?}", y);

    println!("Testing C vector of matrices");

    let aref = vec![1., 2.];

    let a = CsrMatrix {
        x: aref.as_slice().as_ptr(),
        size: 2,
        a: 1
    };

    let b = CsrMatrix {
        x: [3., 4.].as_ptr(),
        size: 2,
        a: 2
    };

    let mut v = vec![a, b];

    list_of_m_ffi(&mut v[..]);

}
