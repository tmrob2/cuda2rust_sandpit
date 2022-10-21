# cuda2rust_sandpit

Simplest configuration and testing of Rust sparse matrix
data structures for sparse matrix linear algebra on CUDA. 

First, define a `build.rs` which configures the linking of
cuda libs necessary for computation. Then `C` libraries are
created under `myclib` which creates the most basic interface for controlling cuBlas and cuSparse linalg routines. 

A very simple FFI is constructed in `lib.rs` which references functions in `myclib`, and `build.rs` manages the construction of the dynamic shared library using:
```Rust
cc::Build::new()
        .file("myclib/mycfuncs.c")
        .file("myclib/test_cublas.c")
        .file("myclib/test_cusparse.c")
        .file("myclib/sparse_funcs.c")
        .cuda(true)
        .compile("mycfuncs");
```

The idea is to manage memory using Rust, bue still leverage the low level power of CUDA which is most conveniently accessed via a custom `C` library. 

## Why not use `cuBlas` FFI?

I don't see any point in using a monolithic FFI for CUDA libs when it is clear that CUDA interfaces will not be natively written in `Rust` any time soon. Further, there is no FFI for `cuSparse` which is arguably the more important CUDA library. 

## Implementation Notes

A quick note on `build.rs` linking ordering. For `cargo test` to work the linked libraries need to go after `cc::Build::new()` otherwise a linking error will be generated, i.e.:

```Rust
cc::Build::new()
    .file("myclib/mycfuncs.c")
    .file("myclib/test_cublas.c")
    .file("myclib/test_cusparse.c")
    .cuda(true)
    .compile("mycfuncs");
    
println!("cargo:rustc-link-lib=dylib=cublas");
```

### How to start a cuSparse session from `Rust`

On the `Rust` side in `lib.rs` we require some way of creating a `cusparseHandle_t`. Similarly, this can be achieved in the same way for `cuBlas`. This is done using the following functions.

Creation of `handle` requires the pair:
```Rust
pub struct cusparseContext {
    _unused: [u8; 0],
}

pub type cusparseHandle_t = *mut cusparseContext;

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
```

Naturally, we also need to free the handle from memory to avoid memory leaks on the device side (GPU). This is done using the following functions. 
```Rust
extern "C" {
    pub fn destroy_session(handle: *mut cusparseHandle_t);
}

pub fn destroy_session_ffi(handle: *mut cusparseHandle_t) {
    unsafe {
        destroy_session(handle);
    }
}
```

