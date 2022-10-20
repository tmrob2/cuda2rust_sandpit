use rand::Rng;
use ctest::{doubler_interface, cuda_spaxy_ffi};

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
    cuda_spaxy_ffi(&x[..], &mut y[..], alpha)

}
