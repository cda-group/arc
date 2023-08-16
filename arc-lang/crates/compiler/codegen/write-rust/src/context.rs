// use im_rc::Vector;
//
// pub struct Context {
//     pub ir: mir::IR,
//     functions: Vec<syn::ItemFn>,
//     structs: Vec<syn::ItemStruct>,
//     enums: Vec<syn::ItemEnum>,
//
//     function_ids: OrdMap<(mir::Name, Vector<mir::Type>), syn::Ident>,
//     struct_ids: OrdMap<mir::Type, syn::Ident>,
//     enum_ids: OrdMap<(mir::Name, Vector<mir::Type>), syn::Ident>,
// }
//
// pub fn id(x: impl AsRef<str>) -> syn::Ident {
//     syn::Ident::new(x.as_ref(), proc_macro2::Span::call_site())
// }
//
// impl Context {
//     pub fn new(ir: mir::IR) -> Self {
//         Context {
//             ir,
//             functions: Vec::new(),
//             structs: Vec::new(),
//             enums: Vec::new(),
//             function_ids: OrdMap::new(),
//             struct_ids: OrdMap::new(),
//             enum_ids: OrdMap::new(),
//         }
//     }
//
//     pub fn add_function(mut self, item: syn::ItemFn) -> Self {
//         self.functions.push(item);
//         self
//     }
//
//     pub fn add_struct(mut self, item: syn::ItemStruct) -> Self {
//         self.structs.push(item);
//         self
//     }
//
//     pub fn add_enum(mut self, item: syn::ItemEnum) -> Self {
//         self.enums.push(item);
//         self
//     }
//
//     pub fn has_function(&self, x: &mir::Name, ts: &Vector<mir::Type>) -> bool {
//         self.function_ids.contains_key(&(x.clone(), ts.clone()))
//     }
//
//     pub fn has_struct(&self, id: &mir::Type) -> bool {
//         self.struct_ids.contains_key(id)
//     }
//
//     pub fn has_enum(&self, id: &(mir::Name, Vector<mir::Type>)) -> bool {
//         self.enum_ids.contains_key(id)
//     }
//
//     pub fn get_function_id(mut self, x: mir::Name, ts: Vector<mir::Type>) -> (Self, Ident) {
//         todo!()
//         //let len = self.function_ids.len();
//         //let x = self
//         //    .function_ids
//         //    .entry((x.clone(), ts))
//         //    .or_insert_with(|| id(format!("{x}{len}", x = x.last().unwrap())))
//         //    .clone();
//         //(self, x)
//     }
//
//     pub fn get_struct_id(mut self, t: mir::Type) -> (Self, Ident) {
//         let len = self.struct_ids.len();
//         let x = self
//             .struct_ids
//             .entry(t)
//             .or_insert_with(|| syn::Ident::new(&format!("S{len}"), Span::call_site()))
//             .clone();
//         (self, x)
//     }
//
//     pub fn get_enum_id(mut self, x: mir::Name, ts: Vector<mir::Type>) -> (Self, Ident) {
//         let len = self.enum_ids.len();
//         let x = self
//             .enum_ids
//             .entry((x.clone(), ts))
//             .or_insert_with(|| syn::Ident::new(&format!("E{len}"), Span::call_site()))
//             .clone();
//         (self, x)
//     }
//
//     pub fn get_func(&self, x: &mir::Name, ts: &Vector<mir::Type>) -> mir::ItemFunc {
//         self.ir
//             .functions
//             .get(&(x.clone(), ts.clone()))
//             .unwrap()
//             .clone()
//     }
//
//     pub fn get_type(&self, x: &mir::Name, ts: &Vector<mir::Type>) -> mir::ItemType {
//         self.ir
//             .types
//             .get(&(x.clone(), ts.clone()))
//             .unwrap()
//             .clone()
//     }
//
//     pub fn consume(self) -> Rust {
//         Rust {
//             functions: self.functions,
//             structs: self.structs,
//             enums: self.enums,
//         }
//     }
// }
