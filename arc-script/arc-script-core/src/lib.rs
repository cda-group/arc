//! Root module of the compiler.
//!
//! Here we configure crate-wide lints. By default all lints are disallowed to happen.
//!
//! See all lints [here](https://doc.rust-lang.org/rustc/lints/groups.html).
//! Or by running `rustc -W help`

// Deny all clippy lints
#![deny(clippy::correctness)]
#![deny(clippy::style)]
#![deny(clippy::complexity)]
#![deny(clippy::perf)]
#![deny(clippy::pedantic)]
#![deny(clippy::nursery)]
#![deny(clippy::cargo)]
// Exceptions
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::similar_names)]
#![allow(clippy::redundant_else)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::non_ascii_literal)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::redundant_pub_crate)] // Buggy
#![allow(clippy::enum_glob_use)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::shadow_unrelated)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::default_trait_access)] // Breaks `crepe`
#![allow(clippy::match_same_arms)] // Breaks macros
#![allow(clippy::explicit_iter_loop)] // Breaks `crepe`
#![allow(clippy::map_unwrap_or)] // Buggy
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cargo_common_metadata)] // Needless information
#![allow(clippy::option_if_let_else)] // Buggy
#![allow(clippy::module_name_repetitions)] // Annoying
// Allow unused items, toggle this every once in a while
#![allow(unused)]
// Deny all rustc/rustdoc lints (except for a few)
#![deny(absolute_paths_not_starting_with_crate)]
#![deny(anonymous_parameters)]
#![allow(box_pointers)] // Does not make any sense to deny, Box<T> is necessary
#![deny(deprecated_in_future)]
#![deny(elided_lifetimes_in_paths)]
#![deny(explicit_outlives_requirements)]
#![deny(invalid_html_tags)]
#![deny(keyword_idents)]
#![deny(macro_use_extern_crate)]
#![deny(meta_variable_misuse)]
#![deny(missing_abi)]
#![allow(missing_copy_implementations)] // In some cases, we do not want Copy nor Clone
#![deny(missing_crate_level_docs)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(missing_doc_code_examples)]
#![deny(non_ascii_idents)]
#![deny(pointer_structural_match)]
#![deny(private_doc_tests)]
#![allow(single_use_lifetimes)] // Results in bugs when enabled
#![deny(trivial_casts)]
#![deny(trivial_numeric_casts)]
#![deny(unaligned_references)]
#![deny(unreachable_pub)]
#![deny(unsafe_code)]
#![deny(unstable_features)]
#![allow(unused_crate_dependencies)]
#![deny(unused_extern_crates)]
#![deny(unused_import_braces)]
#![deny(unused_lifetimes)]
#![allow(unused_qualifications)] // Buggy
#![allow(unused_results)] // Just very annoying, it means every return value must be used
#![allow(variant_size_differences)] // Good lint, annoying but can be useful for tuning performance
#![deny(warnings)]

/// Module which assembles the compilation pipeline.
pub mod compiler;
/// Module which re-exports common functionality.
pub mod prelude;
