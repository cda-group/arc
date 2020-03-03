//===- Dialect definition for the Rust IR --------------------------------===//
//
// Copyright 2019 The MLIR Authors.
// Copyright 2019 KTH Royal Institute of Technology.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Defines the types of the Rust dialect.
//
//===----------------------------------------------------------------------===//

#ifndef RUST_PRINTER_STREAM_H_
#define RUST_PRINTER_STREAM_H_

#include "rust/Types.h"
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace rust {
// This class main purpose is to track MLIR Values and map them to
// unique ids.
class RustPrinterStream {
  llvm::raw_ostream &OS;

  llvm::raw_string_ostream Constants, Body;

  std::string ConstantsStr, BodyStr;

  unsigned NextID;
  DenseMap<Value, unsigned> Value2ID;

public:
  RustPrinterStream(llvm::raw_ostream &os)
      : OS(os), Constants(ConstantsStr), Body(BodyStr), NextID(0){};

  void flush() {
    OS << Constants.str();
    OS << Body.str();
  }

  llvm::raw_ostream &getBodyStream() { return Body; }

  llvm::raw_ostream &getConstantsStream() { return Constants; }

  unsigned get(Value v) {
    if (Value2ID.find(v) == Value2ID.end())
      Value2ID[v] = NextID++;
    return Value2ID[v];
  }

  unsigned getConstant(RustConstantOp v) {
    if (Value2ID.find(v) == Value2ID.end()) {
      StringAttr str = v.getValue().dyn_cast<StringAttr>();
      unsigned id = get(v);
      types::RustType cType = v.getType().cast<types::RustType>();
      Constants << "const v" << id << " : ";
      cType.printAsRust(Constants) << " = " << str.getValue() << ";\n";
      return id;
    }
    return Value2ID[v];
  }

  RustPrinterStream &print(Value v) {
    Body << "v" << get(v);
    return *this;
  }

  RustPrinterStream &print(types::RustType t) {
    t.printAsRust(Body);
    return *this;
  }

  template <typename T>
  RustPrinterStream &print(T t) {
    Body << t;
    return *this;
  }
};

RustPrinterStream &operator<<(RustPrinterStream &os, const Value &v);

RustPrinterStream &operator<<(RustPrinterStream &os, const types::RustType &t);

template <typename T>
RustPrinterStream &operator<<(RustPrinterStream &os, const T &t) {
  return os.print(t);
}

} // namespace rust

#endif // RUST_PRINTER_STREAM_H_
