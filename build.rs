extern crate cc;

fn main() {

    // Need to include the local install of CUDA tools
    cc::Build::new()
    .file("myclib/mycfuncs.c")
    .file("myclib/test_cublas.c")
    .file("myclib/test_cusparse.c")
    .cuda(true)
    .compile("mycfuncs");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cusparse");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/");
}