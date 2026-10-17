//! mimalloc beyond Rust's own allocations, under the `mimalloc` feature;
//! without it both functions do nothing.

#![cfg_attr(
    feature = "mimalloc",
    expect(unsafe_code, reason = "mimalloc and meshoptimizer are C libraries")
)]

/// Sends meshoptimizer's scratch allocations to mimalloc. It allocates through
/// `operator new`, which reaches the CRT heap rather than Rust's global
/// allocator. Must run before anything calls into meshopt.
pub fn route_meshopt() {
    // SAFETY: nothing has called into meshopt yet, so no meshopt call runs
    // concurrently and nothing it allocated is outstanding; `mi_malloc` and
    // `mi_free` are a matching pair with the C signatures it expects.
    #[cfg(feature = "mimalloc")]
    unsafe {
        meshopt::ffi::meshopt_setAllocator(
            Some(libmimalloc_sys::mi_malloc),
            Some(libmimalloc_sys::mi_free),
        );
    }
}

/// Returns the memory mimalloc holds free to the OS, such as the bake's once
/// its threads have exited.
pub fn collect() {
    // SAFETY: `mi_collect` takes no pointers and may run on any thread.
    #[cfg(feature = "mimalloc")]
    unsafe {
        libmimalloc_sys::mi_collect(true);
    }
}
