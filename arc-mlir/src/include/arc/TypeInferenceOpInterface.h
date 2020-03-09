//===- TypeInferenceInterface.h - Interface definitions for TypeInference -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the type inference interfaces defined
// in TypeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_TYPE_INFERENCE_OP_INTERFACE_H_
#define ARC_TYPE_INFERENCE_OP_INTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace arc {

#include "arc/TypeInferenceOpInterface.h.inc"

} // end namespace arc
} // end namespace mlir

#endif // ARC_TYPE_INFERENCE_OP_INTERFACE_H_
