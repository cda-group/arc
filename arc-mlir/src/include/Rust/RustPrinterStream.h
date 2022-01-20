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

  std::map<std::string, std::string> CrateDependencies;
  std::map<std::string, std::string> CrateDirectives;

  DenseMap<Value, std::string> ValueAliases;
  DenseMap<Type, std::string> TypeAliases;

  std::string ModuleName;
  std::string Includefile;

public:
  RustPrinterStream(std::string moduleName, std::string includefile)
      : Constants(ConstantsStr), NamedTypes(NamedTypesStr), TypeUses(UsesStr),
        Body(BodyStr), NextID(0), NextConstID(0), ModuleName(moduleName),
        Includefile(includefile){};

  void flush(llvm::raw_ostream &o) {
    o << "#[allow(non_snake_case)]\n"
      << "#[allow(unused_must_use)]\n"
      << "#[allow(dead_code)]\n"
      << "#[allow(unused_variables)]\n"
      << "#[allow(unused_imports)]\n"
      << "#[allow(unused_braces)]\n";

    o << "pub mod " << ModuleName
      << "{\n"
         "use super::*;\n"
      << "pub use arc_script::codegen::*;\n"
      << "pub use arc_script::codegen;\n"
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
    else
      return "val!(v" + std::to_string(id) + ")";
  }

  std::string getConstant(RustConstantOp v) {
    StringAttr str = v.getValue().dyn_cast<StringAttr>();
    if (FunctionType fType = v.getType().dyn_cast<FunctionType>()) {
      // Although a function reference is a constant in MLIR it is not
      // in our Rust dialect, so we need to handle them specially.
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

  std::string getTypeString(Type t) {
    auto alias = TypeAliases.find(t);
    if (alias == TypeAliases.end())
      return rust::types::getTypeString(t);
    return alias->second;
  }

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

  RustPrinterStream &print(llvm::raw_ostream &o, types::RustType t) {
    if (printTypeAlias(o, t))
      return *this;
    t.printAsRust(o);
    return *this;
  }

  RustPrinterStream &print(llvm::raw_ostream &o, types::RustEnumType t) {
    if (printTypeAlias(o, t))
      return *this;
    writeEnumDefiniton(t);
    t.printAsRustNamedType(o);
    return *this;
  }

  RustPrinterStream &print(llvm::raw_ostream &o, types::RustStructType t) {
    if (printTypeAlias(o, t))
      return *this;
    writeStructDefiniton(t);
    t.printAsRustNamedType(o);
    return *this;
  }

  RustPrinterStream &print(llvm::raw_ostream &o, types::RustStreamType t) {
    if (printTypeAlias(o, t))
      return *this;
    printAsRust(o, t);
    return *this;
  }

  RustPrinterStream &print(llvm::raw_ostream &o, types::RustSinkStreamType t) {
    if (printTypeAlias(o, t))
      return *this;
    printAsRust(o, t);
    return *this;
  }

  RustPrinterStream &print(llvm::raw_ostream &o,
                           types::RustSourceStreamType t) {
    if (printTypeAlias(o, t))
      return *this;
    printAsRust(o, t);
    return *this;
  }

  RustPrinterStream &print(llvm::raw_ostream &o, FunctionType t) {
    printAsRust(o, t);
    return *this;
  }

  void writeEnumDefiniton(types::RustEnumType t) {
    unsigned id = t.getEnumTypeId();

    // Only output an enum definition once
    if (OutputEnumTypes.find(id) == OutputEnumTypes.end()) {
      OutputEnumTypes.insert(id);
      t.printAsRust(*this);
    }
  }

  void writeStructDefiniton(types::RustStructType t) {
    unsigned id = t.getStructTypeId();

    // Only output a struct definition once
    if (OutputStructTypes.find(id) == OutputStructTypes.end()) {
      OutputStructTypes.insert(id);
      t.printAsRust(*this);
    }
  }

  template <typename T>
  RustPrinterStream &print(llvm::raw_ostream &o, T t) {
    o << t;
    return *this;
  }

  void registerDependency(std::string key, std::string value) {
    CrateDependencies[key] = value;
  }

  void registerDirective(std::string key, std::string value) {
    CrateDirectives[key] = value;
  }

  llvm::raw_ostream &printAsRust(llvm::raw_ostream &s, const Type ty) {
    if (FunctionType fType = ty.dyn_cast<FunctionType>()) {
      s << "Box<dyn ValueFn(";
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
    if (types::RustEnumType rt = ty.dyn_cast<types::RustEnumType>()) {
      this->print(s, rt);
      return s;
    }
    if (types::RustStreamType rt = ty.dyn_cast<types::RustStreamType>()) {
      s << "Stream<<";
      printAsRust(s, rt.getType());
      s << " as Convert>::T>";
      return s;
    }
    if (types::RustSinkStreamType rt =
            ty.dyn_cast<types::RustSinkStreamType>()) {
      s << "Pushable<";
      printAsRust(s, rt.getType());
      s << ">";
      return s;
    }
    if (types::RustSourceStreamType rt =
            ty.dyn_cast<types::RustSourceStreamType>()) {
      s << "Pullable<";
      printAsRust(s, rt.getType());
      s << ">";
      return s;
    }
    if (types::RustStructType rt = ty.dyn_cast<types::RustStructType>()) {
      this->print(s, rt);
      return s;
    }
    if (types::RustTensorType rt = ty.dyn_cast<types::RustTensorType>()) {
      rt.printAsRust(*this);
      return s;
    }
    if (types::RustTupleType rt = ty.dyn_cast<types::RustTupleType>()) {
      rt.printAsRust(*this);
      return s;
    }
    s << "unhandled type";
    return s;
  }

private:
  bool printTypeAlias(llvm::raw_ostream &o, Type t) {
    auto alias = TypeAliases.find(t);
    if (alias == TypeAliases.end())
      return false;
    o << alias->second;
    return true;
  }
};

RustPrinterStream &operator<<(RustPrinterStream &os, const Value &v);

RustPrinterStream &operator<<(RustPrinterStream &os, const types::RustType &t);

template <typename T>
RustPrinterStream &operator<<(RustPrinterStream &os, const T &t) {
  return os.print(os.getBodyStream(), t);
}

} // namespace rust

#endif // RUST_PRINTER_STREAM_H_
