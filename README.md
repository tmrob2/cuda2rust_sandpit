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

It is much easier to write a small implementation using the `cuSparse` header and then create a small FFI related to a `Rust` project, which is playing to the strengths of both languages.

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

## Tests

Run lib tests with `cargo test`.

