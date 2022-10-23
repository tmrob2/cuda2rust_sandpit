/*
**********************************************************
**********************************************************

Section 1: Sparse matrix formation utility functions:
Structures:
1. creation of the CS_SI -> a single precision (f32) sparse
matrix format designed for inputting into cuSparse

Matrix type is specified according to:
CS_SI.sptype -> {SP_TYPE::Triple, SP_TYPE::CSR}

Implementations:
makes a triple as the base function for convenient 
coordinate format.

triple_entry() -> is a function to add entries to the 
CS_SI Triple

**********************************************************
**********************************************************
*/

use std::ffi::c_int;

#[derive(Debug)]
pub enum SP_TYPE {
    Triple,
    CSR
}

#[derive(Debug)]
pub struct CS_SI {
    pub nzmax: i32,
    pub m: i32,
    pub n: i32,
    pub p: Vec<i32>,
    pub i: Vec<i32>,
    pub x: Vec<f32>,
    pub nz: i32,
    pub sptype: SP_TYPE
}

impl CS_SI {
    pub fn make(nzmax: i32, m: i32, n: i32, nz: i32, sptype: SP_TYPE) -> Self {
        // makes a new triple
        CS_SI { 
            nzmax, 
            m, 
            n, 
            p: Vec::new(), 
            i: Vec::new(), 
            x: Vec::new(), 
            nz, 
            sptype
        }
    }

    pub fn triple_entry(&mut self, i: i32, j: i32, val: f32) {
        if self.nz < self.nzmax { 
            self.i.push(i);
            self.p.push(j);
            self.x.push(val);
            self.nz += 1;
        } else {
            println!("entry greater than max entries.");
        }
    }

    pub fn free(&mut self) {
        self.p = Vec::new();
        self.i = Vec::new();
        self.x = Vec::new();
    }

    pub fn csr_compress(&self) -> Self {
        // make a new CS matrix in the CSR format
        let nz = self.nz;
        let mut csr_val: Vec<f32> = vec![0.; self.nz as usize];
        let mut csr_j: Vec<i32> = vec![0; self.nz as usize];
        let mut csr_i: Vec<i32> = vec![0; self.m as usize + 1];
        let nzmax = self.nzmax;
        let m = self.m;
        let n = self.n;
        for k in 0..nz as usize {
            csr_val[k] = self.x[k];
            csr_j[k] = self.p[k];
            csr_i[(self.i[k]) as usize + 1] += 1;
        }
        for r in 0..self.m as usize {
            csr_i[r + 1] += csr_i[r];
        }
        
        CS_SI { 
            nzmax, 
            m, 
            n, 
            p: csr_j, 
            i: csr_i, 
            x: csr_val, 
            nz, 
            sptype: SP_TYPE::CSR 
        }
    }
}

/*
**********************************************************
**********************************************************

C FFI


**********************************************************
**********************************************************
*/

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseContext {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct cublasContext {
    _unused: [u8; 0],
}

pub type cusparseHandle_t = *mut cusparseContext;
pub type cusparseSpMatDescr_t = *mut cusparseContext;
pub type cusparseDnVecDescr_t = *mut cusparseContext;

pub type cublasHandle_t = *mut cublasContext;

extern "C" {
    pub fn create_session(handle: *mut cusparseHandle_t);
}

pub fn create_session_ffi() -> *mut cusparseHandle_t {
    let mut handle: *mut cusparseHandle_t = std::ptr::null_mut();
    unsafe {
        create_session(handle);
    }
    handle
}

extern "C" {
    pub fn destroy_session(handle: *mut cusparseHandle_t);
}

pub fn destroy_session_ffi(handle: *mut cusparseHandle_t) {
    unsafe {
        destroy_session(handle);
    }
}

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

#[link(name="cudatest", kind="static")]
extern "C" {
    fn cuda_call_spaxy(x: *const f32, y: *mut f32, n: i32, alpha: f32);
}

pub fn cuda_spaxy_ffi(x: &[f32], y: &mut [f32], alpha: f32) {
    unsafe {
        cuda_call_spaxy(x.as_ptr(), y.as_mut_ptr(), x.len() as i32, alpha);
    }
}

#[link(name="cudatest", kind="static")]
extern "C" {
    fn csr_spmv(
        csr_row: *const i32, 
        csr_col: *const i32, 
        csr_vals: *const f32,
        x: *const f32,
        y: *mut f32,
        nnz: i32, 
        sizeof_row: i32, 
        m: i32,
        n: i32
    );
}

pub fn csr_spmv_ffi(
    row: &[i32],
    col: &[i32],
    vals: &[f32],
    x: &[f32],
    y: &mut [f32],
    nnz: i32,
    sizeof_row: i32,
    m: i32,
    n: i32,
) {
    unsafe {
        csr_spmv(
            row.as_ptr(), 
            col.as_ptr(), 
            vals.as_ptr(), 
            x.as_ptr(), 
            y.as_mut_ptr(),
            nnz, 
            row.len() as i32,
            m, 
            n
        )
    }
}

use std::ffi;

#[repr(C)]
#[derive(Clone)]
pub struct CsrMatrix {
    pub x: *const f32,
    pub size: i32,
    pub a: i32
}

extern "C" {
    fn my_c_struct_ptr(p: *mut CsrMatrix, len: libc::size_t);
}

pub fn list_of_m_ffi(v: &mut [CsrMatrix]) {
    unsafe {
        my_c_struct_ptr(v.as_mut_ptr(), v.len() as libc::size_t);
    }
}

#[link(name="cudatest", kind="static")]
extern "C" {
    fn call_reshape(v: *const f32, m: *mut f32, size_x: i32, num_actions: i32);
}

pub fn call_reshape_ffi(v: &[f32], m: &mut [f32], size_x: i32, num_actions: i32) {
    unsafe {
        call_reshape(v.as_ptr(), m.as_mut_ptr(), size_x, num_actions);
    }
}


/*
**********************************************************
**********************************************************

Tests


**********************************************************
**********************************************************
*/

#[cfg(test)]
mod tests {
    use rand::Rng;
    use crate::{cadd, CS_SI, cuda_spaxy_ffi, SP_TYPE, 
        create_session, create_session_ffi, destroy_session_ffi,
        cusparseSpMatDescr_t, csr_spmv_ffi, call_reshape_ffi};//, cuda_spaxy_ffi};
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

        cuda_spaxy_ffi(&x, &mut y, alpha);
        assert_eq!(y, y_test)
    }

    #[test]
    fn test_create_new_sp_matrix() {
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
    }

    #[test]
    fn coo_to_csr_test_1() {
        // create a new sparse matrix in COO format
        let row: Vec<i32> = vec![0, 1, 2, 3];
        let col: Vec<i32> = vec![0, 1, 2, 1];
        let vals: Vec<f32> = vec![5., 8., 3., 6.];
        let nzmax = 4; 
        let nz = 0;
        let m = 4;
        let n = 4;
        let mut coo = CS_SI::make(nzmax, m, n, nz, SP_TYPE::Triple);
        for k in 0..vals.len() as usize {
            coo.triple_entry(row[k], col[k], vals[k]);
        }
        let csr = coo.csr_compress();

        assert_eq!(csr.x, coo.x);
        assert_eq!(csr.p, coo.p);
        assert_eq!(csr.i, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn coo_to_csr_test_2() {
        let row: Vec<i32> = vec![0, 0, 1, 1, 2, 2, 2, 3];
        let col: Vec<i32> = vec![0, 1, 1, 3, 2, 3, 4, 5]; 
        let val: Vec<f32> = vec![10., 20., 30., 40., 50., 60., 70., 80.];
        let nzmax = val.len() as i32;
        let nz = 0;
        let m = 4;
        let n = 6;
        let mut coo = CS_SI::make(nzmax, m, n, nz, SP_TYPE::Triple);
        for k in 0..val.len() {
            coo.triple_entry(row[k], col[k], val[k]);
        }
        let csr = coo.csr_compress();
        assert_eq!(csr.x, val);
        assert_eq!(csr.p, col);
        assert_eq!(csr.i, vec![0, 2, 4, 7, 8])
    }

    #[test]
    fn cusparse_csr_input() {
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

        assert_eq!(y, vec![19.0, 8.0, 51.0, 52.0]);

        //destroy_session_ffi(handle);
    }

    #[test]
    fn cuda_reshape() {

        let num_actions = 4;
        let v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        
        let mut m: Vec<f32> = vec![0.; v.len() * num_actions];

        call_reshape_ffi(&v, &mut m, v.len() as i32, num_actions as i32);

        assert_eq!(m, 
            vec![
                1.0, 2.0, 3.0, 4.0, 
                1.0, 2.0, 3.0, 4.0, 
                1.0, 2.0, 3.0, 4.0, 
                1.0, 2.0, 3.0, 4.0
                ]);
    }
}

