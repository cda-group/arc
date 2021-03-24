use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::FunKind::Global;
use crate::compiler::hir::Name;
use crate::compiler::hir::Path;
use crate::compiler::info;
use crate::compiler::info::diags::Error;
use crate::compiler::info::files::Loc;
use crate::compiler::info::types::TypeId;

use arc_script_core_shared::map;
use arc_script_core_shared::Lower;
use arc_script_core_shared::New;
use arc_script_core_shared::VecMap;

use super::Context;
