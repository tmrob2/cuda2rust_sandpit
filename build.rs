extern crate cc;

fn main() {

    // Need to include the local install of CUDA tools
    cc::Build::new()
        .file("myclib/mycfuncs.c")
        .file("myclib/test_array_csr.c")
        .compile("mycfuncs");

    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_86,code=sm_86")
        .file("myclib/test_cusparse.cu")
        .file("myclib/test_cublas.cu")
        .compile("libcudatest.a");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/");
    //println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cusparse");
}