all:
	clang -c mlir-src/hello.c \
		    -o mlir-target/hello.o && \
	llvm-ar rc \
	      mlir-target/libhello.a \
	      mlir-target/hello.o

clean:
	rm mlir-target/libhello.a \
	   mlir-target/hello.o
