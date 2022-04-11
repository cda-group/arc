all:
	cargo doc --document-private-items --no-deps --target-dir=src/target --manifest-path=../arc-runtime/Cargo.toml
	cargo doc --document-private-items --no-deps --target-dir=src/target --manifest-path=../arc-runtime/macros/Cargo.toml
	mdbook build

build:
	mdbook build
