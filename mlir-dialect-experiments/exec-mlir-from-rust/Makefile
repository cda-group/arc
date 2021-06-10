AR=../../arc-mlir/build/llvm-build/bin/llvm-ar
ARFLAGS=rc

CC=../../arc-mlir/build/llvm-build/bin/llc
CCFLAGS=-filetype=obj

MLIR_OPT=../../arc-mlir/build/llvm-build/bin/mlir-opt
MLIR_OPT_FLAGS=--gpu-to-llvm --convert-std-to-llvm --llvm-legalize-for-export

MLIR_TRANSLATE=../../arc-mlir/build/llvm-build/bin/mlir-translate
MLIR_TRANSLATE_FLAGS=--mlir-to-llvmir

SRC=mlir/src/hello.mlir
OBJ=mlir/target/hello.o
IR=mlir/target/hello.ll
LIB=mlir/target/libhello.a

all: $(LIB)

$(LIB): $(OBJ)
	$(AR) $(ARFLAGS) $@ $^

$(OBJ): $(IR)
	$(CC) $(CCFLAGS) -o=$@ $^

$(IR): $(SRC)
	$(MLIR_OPT) $(MLIR_OPT_FLAGS) $^ | \
	$(MLIR_TRANSLATE) $(MLIR_TRANSLATE_FLAGS) -o=$@

clean:
	$(RM) $(LIB) $(OBJ) $(IR)
