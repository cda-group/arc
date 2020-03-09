//===- TypeInferenceInterface.td - Type Inference Interface -*- tablegen --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Type Inference Op Interface.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_TYPE_INFERENCE_INTERFACE
#define ARC_TYPE_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

def TypeInferenceOpInterface : OpInterface<"TypeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the type for the current operation.",
                    "void", "inferTypes">
  ];
}

#endif // ARC_TYPE_INFERENCE_INTERFACE
