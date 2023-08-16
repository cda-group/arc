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

namespace rust {
class RustPrinterStream;
}

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
  llvm::raw_string_ostream Constants, NamedTypes, TypeUses, Body;

  std::string ConstantsStr, NamedTypesStr, UsesStr, BodyStr;

  // Variables are using positive IDs, constants are assigned negative
  // IDs.
  int NextID, NextConstID;
  DenseMap<Value, int> Value2ID;

  // Tracking of the named types which has already been output.
  DenseSet<unsigned> OutputEnumTypes, OutputStructTypes;

  // Functions having the rust.declare attribute
  DenseSet<Operation *> DeclaredFunctions;
  // Tasks to declare, triggered by the arc.is_task attribute
  DenseSet<RustFuncOp> DeclaredTasks;

  std::map<std::string, std::string> CrateDependencies;
  std::map<std::string, std::string> CrateDirectives;

  DenseMap<Value, std::string> ValueAliases;

  // This maps function types to their mangled names. In contrast to
  // the other types, which caches the mangled name in their type
  // storage class, we have to maintain the mapping in the
  // RustPrinterStream as we don't want to extend the upstream class.
  DenseMap<Type, std::string> FunctionTypes;

  std::string ModuleName;
  std::string Includefile;

public:
  RustPrinterStream(std::string moduleName, std::string includefile)
      : Constants(ConstantsStr), NamedTypes(NamedTypesStr), TypeUses(UsesStr),
        Body(BodyStr), NextID(0), NextConstID(0), ModuleName(moduleName),
        Includefile(includefile){};

  void flush(llvm::raw_ostream &o) {
    o << "#[allow(unused)]\n"
         "#[allow(non_snake_case)]\n"
         "#[allow(non_camel_case_types)]\n";

    o << "pub mod " << ModuleName
      << "{\n"
         "use super::*;\n"
         "pub use runtime::prelude::*;\n";

    for (auto i : CrateDirectives)
      o << i.second << "\n";
    o << Constants.str();
    o << TypeUses.str();
    o << NamedTypes.str();
    o << Body.str();
    if (!Includefile.empty())
      o << "include!(\"" << Includefile << "\");\n";
    o << "}\n";
    o << "pub use " << ModuleName << "::*;\n";
  }

  // Returns true if there has been output to the types NamedTypes
  // stream.
  bool hasTypesOutput() const { return !NamedTypesStr.empty(); };

  llvm::raw_ostream &getBodyStream() { return Body; }

  llvm::raw_ostream &getNamedTypesStream() { return NamedTypes; }

  llvm::raw_ostream &getUsesStream() { return TypeUses; }

  llvm::raw_ostream &getConstantsStream() { return Constants; }

  void addAlias(Value v, std::string identifier) {
    ValueAliases[v] = identifier;
  }

  void addTask(RustFuncOp &task) { DeclaredTasks.insert(task); }
  void addDeclaredFunction(Operation *f) { DeclaredFunctions.insert(f); }

  void clearAliases() { ValueAliases.clear(); }

  std::string get(Value v) {
    auto alias = ValueAliases.find(v);
    if (alias != ValueAliases.end())
      return alias->second;
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
      return "v" + std::to_string(id) + ".clone()";
  }

  std::string getConstant(RustConstantOp v);

  RustPrinterStream &print(Value v) {
    Body << get(v);
    return *this;
  }

  RustPrinterStream &printAsLValue(Value v) {
    auto alias = ValueAliases.find(v);
    if (alias != ValueAliases.end())
      Body << alias->second;
    auto found = Value2ID.find(v);
    int id = 0;
    if (found == Value2ID.end()) {
      id = NextID++;
      Value2ID[v] = id;
    } else
      id = found->second;
    if (id < 0)
      Body << "C" + std::to_string(-id);
    else
      Body << "v" + std::to_string(id);
    return *this;
  }

  RustPrinterStream &printAsArg(Value v) {
    int id = NextID++;
    Value2ID[v] = id;
    Body << "v" << std::to_string(id);
    return *this;
  }

  void printType(llvm::raw_ostream &o, Type t);

  void registerDependency(std::string key, std::string value) {
    CrateDependencies[key] = value;
  }

  void registerDirective(std::string key, std::string value) {
    CrateDirectives[key] = value;
  }

  std::string getMangledName(FunctionType fTy);
  void printAsRust(llvm::raw_ostream &o, FunctionType fTy);

  RustPrinterStream &let(const Value v);
};

RustPrinterStream &operator<<(RustPrinterStream &os, const Value &v);

RustPrinterStream &operator<<(RustPrinterStream &os, const Type &t);

RustPrinterStream &operator<<(RustPrinterStream &os, const llvm::StringRef &s);

RustPrinterStream &operator<<(RustPrinterStream &os, uint64_t u);

} // namespace rust

#endif // RUST_PRINTER_STREAM_H_
