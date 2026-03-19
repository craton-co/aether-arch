fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_dir = std::path::Path::new(&crate_dir).join("include");
    std::fs::create_dir_all(&output_dir).ok();

    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap_or_default();

    // Fix #18: fail the build if cbindgen cannot generate the header.
    // Previously this was a warning, which could leave a stale aether.h
    // that doesn't match the library — causing ABI mismatches and
    // memory corruption at runtime for C consumers.
    let bindings = cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("cbindgen failed to generate aether.h — fix the FFI definitions");

    bindings.write_to_file(output_dir.join("aether.h"));
}
