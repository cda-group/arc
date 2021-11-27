open Utils

let add_item i ast = i::ast

let add_items ast items = items |> foldl (fun ast i -> ast |> add_item i) ast

(* Types *)

let nominal x ts = Ast.TPath ([x], ts)
let atom x = (x, Ast.TPath ([x], []), Ast.IExternType (x, []))
let generic x = Ast.TPath ([x], [])

let (x_unit, t_unit, i_unit) = atom "unit"
let (x_i16, t_i16, i_i16) = atom "i16"
let (x_i32, t_i32, i_i32) = atom "i32"
let (x_i64, t_i64, i_i64) = atom "i64"
let (x_i128, t_i128, i_i128) = atom "i128"
let (x_f32, t_f32, i_f32) = atom "f32"
let (x_f64, t_f64, i_f64) = atom "f64"
let (x_bool, t_bool, i_bool) = atom "bool"
let (x_char, t_char, i_char) = atom "char"
let (x_str, t_str, i_str) = atom "str"

let scalars = [i_i32; i_unit; i_i16; i_i32; i_i64; i_i128; i_f32; i_f64; i_bool; i_char; i_str]

let ints = [t_i16; t_i32; t_i64; t_i128]
let floats = [t_f32; t_f64]
let nums = ints @ floats

(* Binary operators *)

let binop x t0 t1 t2 = Ast.IExternDef (x, [], [t0; t1], t2)

let insts f ts = ts |> map f

let i_add t = binop "add" t t t_i32
let i_sub t = binop "sub" t t t_i32
let i_mul t = binop "mul" t t t_i32
let i_div t = binop "div" t t t_i32
let i_pow t = binop "pow" t t t_i32
let i_geq t = binop "geq" t t t_bool
let i_leq t = binop "leq" t t t_bool
let i_gt t = binop "gt" t t t_bool
let i_lt t = binop "lt" t t t_bool
let i_mod t = binop "lt" t t t_i32

let i_bor t = binop "bor" t t t
let i_bxor t = binop "bxor" t t t
let i_band t = binop "band" t t t

let i_and _ = binop "and" t_bool t_bool t_bool
let i_or _ = binop "or" t_bool t_bool t_bool
let i_xor _ = binop "xor" t_bool t_bool t_bool

(* Unary operators *)

let unop x t0 t1 = Ast.IExternDef (x, [], [t0], t1)

let i_neg t = unop "neg" t t
let i_not _ = unop "not" t_bool t_bool

(* Scalars *)
and add_scalars ast = ast |> add_items scalars

and add_binops ast =
  ast |> add_items (insts i_add ints)    
      |> add_items (insts i_sub ints)    
      |> add_items (insts i_mul ints)    
      |> add_items (insts i_div ints)    
      |> add_items (insts i_pow ints)    
      |> add_items (insts i_geq ints)   
      |> add_items (insts i_leq ints)   
      |> add_items (insts i_gt ints)    
      |> add_items (insts i_lt ints)    
      |> add_items (insts i_lt ints)     
      |> add_items (insts i_bor ints)    
      |> add_items (insts i_bxor ints)   
      |> add_items (insts i_band ints)   
      |> add_items (insts i_and []) 
      |> add_items (insts i_or [])  
      |> add_items (insts i_xor []) 

and add_unops ast =
  ast |> add_items (insts i_neg [t_i32])

(* Ranges *)

let rec i_range = Ast.IExternType ("Range", ["T"])
and t_range = Ast.TPath (["Range"], [])
and i_range_leq = i_leq 

and add_range ast =
  ast |> add_item i_range

(* Iterators *)

and add_iter ast =
  ast |> add_item (Ast.IExternType ("Iter", ["T"]))
      |> add_item (Ast.IExternDef ("next", ["T"], [(Ast.TPath (["Iter"], [generic "T"]))], generic "T"))

(* Cells *)
and add_cells ast =
  let t_cell = (nominal "Cell" [generic "T"]) in
  let t_elem = generic "T" in
  ast |> add_item (Ast.IExternType ("Cell", ["T"]))
      |> add_item (Ast.IExternDef ("new_cell", ["T"], [t_elem], t_cell))
      |> add_item (Ast.IExternDef ("update_cell", ["T"], [t_cell; t_elem], t_unit))
      |> add_item (Ast.IExternDef ("read_cell", ["T"], [t_cell], t_cell))

(* arraytors *)
and add_arrays ast =
  let t_array = (nominal "Array" [generic "T"]) in
  let t_elem = generic "T" in
  ast |> add_item (Ast.IExternType ("Array", ["T"]))
      |> add_item (Ast.IExternDef ("new_array", ["T"], [t_elem], t_array))
      |> add_item (Ast.IExternDef ("push_array", ["T"], [t_array; t_elem], t_unit))
      |> add_item (Ast.IExternDef ("pop_array", ["T"], [t_array], t_elem))
      |> add_item (Ast.IExternDef ("select_array", ["T"], [t_array; t_i32], t_elem))
      |> add_item (Ast.IExternDef ("len_array", ["T"], [t_array], t_i32))
      |> add_item (Ast.IExternDef ("extend_array", ["T"], [t_array; t_array], t_unit))

and add_streams ast =
  let t_elem = generic "A" in
  let t_elem' = generic "B" in
  let t_stream = (nominal "Stream" [t_elem]) in
  let t_stream' = (nominal "Stream" [t_elem']) in
  let t_fun = Ast.TFunc ([t_elem], t_elem') in
  ast |> add_item (Ast.IExternType ("Stream", ["T"]))
      |> add_item (Ast.IExternDef ("map", ["A"; "B"], [t_stream; t_fun], t_stream'))

(* Intrinsics *)
let add_intrinsics ast =
  ast |> add_scalars
      |> add_cells
      |> add_binops
      |> add_arrays
      |> add_unops
      |> add_iter
      |> add_streams
