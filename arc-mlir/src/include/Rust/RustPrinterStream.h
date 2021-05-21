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
  DenseSet<unsigned> OutputNamedTypes;

  std::map<std::string, std::string> CrateDependencies;
  std::map<std::string, std::string> CrateDirectives;

  DenseMap<Value, std::string> ValueAliases;
  DenseMap<Type, std::string> TypeAliases;

  std::string Includefile;

public:
  RustPrinterStream(std::string includefile)
      : Constants(ConstantsStr), NamedTypes(NamedTypesStr), TypeUses(UsesStr),
        Body(BodyStr), NextID(0), NextConstID(0), Includefile(includefile){};

  void flush(llvm::raw_ostream &o) {
    o << "#[allow(non_snake_case)]\n"
      << "#[allow(unused_must_use)]\n"
      << "#[allow(dead_code)]\n"
      << "#[allow(unused_variables)]\n"
      << "#[allow(unused_imports)]\n"
      << "#[allow(unused_braces)]\n";

    o << "pub mod defs {\n"
         "use super::*;\n"
      << "pub use arc_script::arcorn;\n"
      << "pub use arcon::prelude::*;\n"
      << "pub use hexf::*;\n";

    for (auto i : CrateDirectives)
      o << i.second << "\n";
    o << Constants.str();
    o << TypeUses.str();
    std::string types = NamedTypes.str();
    o << types;
    o << Body.str();
    if (!Includefile.empty())
      o << "include!(\"" << Includefile << "\");\n";
    o << "}\n";
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

  void addAlias(Type t, std::string identifier) { TypeAliases[t] = identifier; }

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
    else {
      Type t = v.getType();
      if (t.isa<FunctionType>() || t.isa<types::RustEnumType>() ||
          t.isa<types::RustStructType>() || t.isa<types::RustTensorType>() ||
          t.isa<types::RustTupleType>())
        return "v" + std::to_string(id) + ".clone()";
      return "v" + std::to_string(id);
    }
  }

  std::string getConstant(RustConstantOp v) {
    StringAttr str = v.getValue().dyn_cast<StringAttr>();
    if (FunctionType fType = v.getType().dyn_cast<FunctionType>()) {
      // Although a function reference is a constant in MLIR it is not
      // in the Arcorn dialect of Rust, so we need to handle them
      // specially.
      auto found = Value2ID.find(v);
      int id = 0;
      if (found == Value2ID.end()) {
        id = NextID++;
        Value2ID[v] = id;
        Body << "let v" << id << " : ";
        printAsRust(Body, fType) << " = Box::new(" << str.getValue() << ") as ";
        printAsRust(Body, fType) << ";\n";
      } else
        id = found->second;
      return "v" + std::to_string(id);
    }
    auto found = Value2ID.find(v);
    int id = 0;
    if (found == Value2ID.end()) {
      id = --NextConstID;
      Value2ID[v] = id;
    } else
      id = found->second;
    types::RustType cType = v.getType().cast<types::RustType>();
    Constants << "const C" << -id << " : ";
    cType.printAsRust(Constants) << " = " << str.getValue() << ";\n";
    return "C" + std::to_string(-id);
  }

  RustPrinterStream &print(Value v) {
    Body << get(v);
    return *this;
  }

  RustPrinterStream &printAsArg(Value v) {
    int id = NextID++;
    Value2ID[v] = id;
    Body << "v" << std::to_string(id);
    return *this;
  }

  RustPrinterStream &print(types::RustType t) {
    if (printTypeAlias(t))
      return *this;
    t.printAsRust(Body);
    return *this;
  }

  RustPrinterStream &print(types::RustEnumType t) {
    if (printTypeAlias(t))
      return *this;
    writeEnumDefiniton(t);
    t.printAsRustNamedType(Body);
    return *this;
  }

  RustPrinterStream &print(types::RustStructType t) {
    if (printTypeAlias(t))
      return *this;
    writeStructDefiniton(t);
    t.printAsRustNamedType(Body);
    return *this;
  }

  void writeEnumDefiniton(types::RustEnumType t) {
    unsigned id = t.getEnumTypeId();

    // Only output an enum definition once
    if (OutputNamedTypes.find(id) == OutputNamedTypes.end()) {
      OutputNamedTypes.insert(id);
      t.printAsRust(*this);
    }
  }

  void writeStructDefiniton(types::RustStructType t) {
    unsigned id = t.getStructTypeId();

    // Only output a struct definition once
    if (OutputNamedTypes.find(id) == OutputNamedTypes.end()) {
      OutputNamedTypes.insert(id);
      t.printAsRust(*this);
    }
  }

  template <typename T>
  RustPrinterStream &print(T t) {
    Body << t;
    return *this;
  }

  void registerDependency(std::string key, std::string value) {
    CrateDependencies[key] = value;
  }

  void registerDirective(std::string key, std::string value) {
    CrateDirectives[key] = value;
  }

  llvm::raw_string_ostream &printAsRust(llvm::raw_string_ostream &s,
                                        const Type ty) {
    if (FunctionType fType = ty.dyn_cast<FunctionType>()) {
      s << "Box<dyn arcorn::ArcornFn(";
      for (Type t : fType.getInputs()) {
        printAsRust(s, t) << ",";
      }
      s << ")";
      if (fType.getNumResults()) {
        s << " -> ";
        printAsRust(s, fType.getResult(0));
      }
      s << ">";
      return s;
    }
    if (types::RustType rt = ty.dyn_cast<types::RustType>()) {
      rt.printAsRust(s);
      return s;
    }
    s << "unhandled type";
    return s;
  }
private:
  bool printTypeAlias(Type t) {
    auto alias = TypeAliases.find(t);
    if (alias == TypeAliases.end())
      return false;
    Body << alias->second;
    return true;
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
