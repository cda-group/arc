all:
	../../arc-mlir/build/llvm-build/bin/mlir-opt \
			--gpu-to-llvm \
			--convert-std-to-llvm \
			--llvm-legalize-for-export \
			mlir-src/hello.mlir | \
	../../arc-mlir/build/llvm-build/bin/mlir-translate \
			--mlir-to-llvmir \
			-o mlir-target/hello.ll && \
	../../arc-mlir/build/llvm-build/bin/llc \
	  	mlir-target/hello.ll \
			-filetype=obj && \
	../../arc-mlir/build/llvm-build/bin/llvm-ar \
			rc \
	  	mlir-target/libhello.a \
	  	mlir-target/hello.o

clean:
	rm mlir-target/*
