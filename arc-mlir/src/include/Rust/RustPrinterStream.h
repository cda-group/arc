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

#include "Rust/Rust.h"
#include "Rust/Types.h"
#include "mlir/IR/Attributes.h"
#include <llvm/Support/raw_ostream.h>

#include <map>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

using namespace mlir;

namespace rust {
// This class main purpose is to track MLIR Values and map them to
// unique ids.
class RustPrinterStream {
  llvm::raw_ostream &OS;

  llvm::raw_string_ostream Constants, Body;

  std::string ConstantsStr, BodyStr;

  // Variables are using positive IDs, constants are assigned negative
  // IDs.
  int NextID, NextConstID;
  DenseMap<Value, int> Value2ID;

  std::map<std::string, std::string> CrateDependencies;
  std::map<std::string, std::string> CrateDirectives;

public:
  RustPrinterStream(llvm::raw_ostream &os)
      : OS(os), Constants(ConstantsStr), Body(BodyStr), NextID(0),
        NextConstID(0){};

  void flush() {
    for (auto i : CrateDirectives)
      OS << i.second << "\n";
    OS << Constants.str();
    OS << Body.str();
  }

  llvm::raw_ostream &getBodyStream() { return Body; }

  llvm::raw_ostream &getConstantsStream() { return Constants; }

  std::string get(Value v) {
    auto found = Value2ID.find(v);
    int id = 0;
    if (found == Value2ID.end()) {
      id = NextID++;
      Value2ID[v] = id;
    } else
      id = found->second;
    if (id < 0)
      return "C" + std::to_string(-id);
    else
      return "v" + std::to_string(id);
  }

  std::string getConstant(RustConstantOp v) {
    auto found = Value2ID.find(v);
    int id = 0;
    if (found == Value2ID.end()) {
      id = --NextConstID;
      Value2ID[v] = id;
    } else
      id = found->second;
    StringAttr str = v.getValue().dyn_cast<StringAttr>();
    types::RustType cType = v.getType().cast<types::RustType>();
    Constants << "const C" << -id << " : ";
    cType.printAsRust(Constants) << " = " << str.getValue() << ";\n";
    return "C" + std::to_string(-id);
  }

  RustPrinterStream &print(Value v) {
    Body << get(v);
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

  void registerDependency(RustDependencyOp dep) {
    std::string key = dep.getCrate().cast<StringAttr>().getValue().str();
    std::string value = dep.getVersion().cast<StringAttr>().getValue().str();
    CrateDependencies[key] = value;
  }

  void writeTomlDependencies(llvm::raw_ostream &out) {
    for (auto i : CrateDependencies)
      out << i.first << " = "
          << "\"" << i.second << "\"\n";
  }

  void registerDirective(RustModuleDirectiveOp dep) {
    std::string key = dep.getKey().cast<StringAttr>().getValue().str();
    std::string str = dep.getStr().cast<StringAttr>().getValue().str();
    CrateDirectives[key] = str;
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
