//====- Passes.cpp - Registration of Arc passes --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Arc/Passes.h"
#include "Arc/Arc.h"

#define GEN_PASS_REGISTRATION
#include "Arc/Passes.h.inc"

void arc::registerArcPasses() { registerPasses(); }
