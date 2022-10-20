extern "C" {
    fn doubler(x: i32) -> i32;
}

pub fn doubler_interface(x: i32) -> i32 {
    unsafe {
        doubler(x)
    }
}

extern "C" {
    fn add(x: *const f32, y: *const f32, z: *mut f32, N: i32);
}

pub fn cadd<'a>(x: &[f32], y: &[f32], z: &mut [f32]) {
    unsafe {
        add(x.as_ptr(), y.as_ptr(), z.as_mut_ptr(), x.len() as i32)
    }
}

extern "C" {
    fn cuda_call_spaxy(x: *const f32, y: *mut f32, n: i32, alpha: f32);
}

pub fn cuda_spaxy_ffi(x: &[f32], y: &mut [f32], alpha: f32) {
    unsafe {
        cuda_call_spaxy(x.as_ptr(), y.as_mut_ptr(), x.len() as i32, alpha);
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use crate::{cadd};//, cuda_spaxy_ffi};
    #[test]
    fn test_cadd() {
        let mut rng = rand::thread_rng();

        let x: Vec<f32> = (1..10).map(|_| rng.gen_range(0.0..10.0)).collect();
        let y: Vec<f32> = (1..10).map(|_| rng.gen_range(0.0..10.0)).collect();
        let z: Vec<f32> = x.iter().enumerate().map(|(i, x)| *x + y[i]).collect();
        let mut z_: Vec<f32> = vec![0.; x.len()];    
        cadd(&x, &y, &mut z_);
        assert_eq!(z, z_)
    }

    #[test]
    fn test_cuda_spaxy() {
        let mut rng = rand::thread_rng();
        let alpha = 1.0;
        let x: Vec<f32> = (1..10).map(|_| rng.gen_range(0.0..10.0)).collect();
        let mut y: Vec<f32> = (1..10).map(|_| rng.gen_range(0.0..10.0)).collect();

        let y_test: Vec<f32> = x.iter().enumerate()
            .map(|(i, x_)| alpha * x_ + y[i]).collect();

        cuda_spaxy_ffi(&x, &mut y);
        assert_eq!(y, y_test)
    }
}

