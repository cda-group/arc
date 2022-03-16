%start <Ast.ast> program
%start <Ast.expr> expr
%%

expr: expr0 Eof { $1 }
program: items Eof { $1 }

(* Utilities *)
%inline paren(x): "(" x ")" { $2 }
%inline brack(x): "[" x "]" { $2 }
%inline brace(x): "{" x "}" { $2 }
%inline angle(x): "<" x ">" { $2 }

%inline epsilon: {}

%inline llist(x):
  llist_rev(x) { $1 |> List.rev }

llist_rev(x):
  | epsilon { [] }
  | llist_rev(x) x { $2::$1 }

%inline nonempty_llist(x):
  nonempty_llist_rev(x) { $1 |> List.rev }

nonempty_llist_rev(x): 
  | x { [$1] }
  | nonempty_llist_rev(x) x { $2::$1 }

%inline separated_nonempty_llist(s, x):
  separated_nonempty_llist_rev(s, x) { $1 |> List.rev }

separated_nonempty_llist_rev(s, x): 
  | x { [$1] }
  | separated_nonempty_llist_rev(s, x) s x { $3::$1 }

%inline separated_llist(s, x):
  | epsilon { [] }
  | separated_nonempty_llist(s, x) { $1 }

%inline separated_llist_trailing(s, x):
  | separated_llist(s, x) s? { $1 }

%inline separated_nonempty_llist_trailing(s, x):
  | separated_nonempty_llist(s, x) s? { $1 }

(* The grammar *)

%inline decorator: "@" brace(separated_llist_trailing(",", field(lit))) { $2 }

%inline items: llist(item) { $1 }
%inline item:
  | loption(decorator) "extern" "def" defname loption(generics) paren(typarams) annot(ty0)? ";"
    { Ast.IExternDef ($1, $4, $5, $6, $7) }
  | loption(decorator) "type" name loption(generics) "=" ty0 ";"
    { Ast.ITypeAlias ($1, $3, $4, $6) }
  | loption(decorator) "extern" "type" name loption(generics) ";"
    { Ast.IExternType ($1, $4, $5) }
  | loption(decorator) "enum" name loption(generics) brace(variants)
    { Ast.IEnum ($1, $3, $4, $5) }
  | loption(decorator) "class" name loption(generics) brace(decls)
    { Ast.IClass ($1, $3, $4, $5) }
  | loption(decorator) "instance" loption(generics) path loption(brack(tys)) brace(defs)
    { Ast.IInstance ($1, $3, $4, $5, $6) }
  | loption(decorator) "def" defname loption(generics) params annot(ty0)? body?
    { Ast.IDef ($1, $3, $4, $5, $6, $7) }
  | loption(decorator) "mod" name brace(items)
    { Ast.IMod ($1, $3, $4) }
  | loption(decorator) "task" defname loption(generics) params ":" params body?
    { Ast.ITask ($1, $3, $4, $5, $7, $8) }
  | loption(decorator) "use" path alias(name)? ";"
    { Ast.IUse ($1, $3, $4) }
  | loption(decorator) "val" name annot(ty0)? "=" expr0 ";"
    { Ast.IVal ($1, $3, $4, $6) }

%inline typarams: separated_llist(",", ty0) { $1 }

%inline decls: llist(decl) { $1 }
%inline decl: "def" name loption(generics) params annot(ty0)?
  { ($2, $3, $4, $5) }

%inline defs: llist(def) { $1 }
%inline def: "def" name loption(generics) params annot(ty0)? block
  { ($2, $3, $4, $5, $6) }

%inline body:
  | "=" expr1 { ([], Some $2) }
  | block { $1 }

%inline annot(x): ":" x { $2 }

%inline params: paren(separated_llist_trailing(",", param)) { $1 }
%inline param: pat0 annot(ty0)? { ($1, $2) }

%inline generics: brack(separated_llist_trailing(",", generic)) { $1 }
%inline generic: Name { $1 }

%inline variants: separated_llist_trailing(",", variant) { $1 }
%inline variant: name loption(paren(tys)) { ($1, $2) }

%inline alias(x): "as" x { $2 }

%inline name: Name { $1 }

%inline binop:
  | ".." { Ast.BRInc }
  | "..=" { Ast.BRExc }
  | "bor" { Ast.BBor }
  | "band" { Ast.BBand }
  | "bxor" { Ast.BBxor }
  | "or" { Ast.BOr }
  | "xor" { Ast.BXor }
  | "and" { Ast.BAnd }
  | "==" { Ast.BEq None }  | "==." { Ast.BEq (Some $1) }
  | "!=" { Ast.BNeq None } | "!=." { Ast.BNeq (Some $1) }
  | "<" { Ast.BLt None }   | "<."  { Ast.BLt (Some $1) }
  | ">" { Ast.BGt None }   | ">."  { Ast.BGt (Some $1) }
  | "<=" { Ast.BLeq None } | "<=." { Ast.BLeq (Some $1) }
  | ">=" { Ast.BGeq None } | ">=." { Ast.BGeq (Some $1) }
  | "+" { Ast.BAdd None }  | "+."  { Ast.BAdd (Some $1) }
  | "-" { Ast.BSub None }  | "-."  { Ast.BSub (Some $1) }
  | "%" { Ast.BMod None }  | "%."  { Ast.BMod (Some $1) }
  | "*" { Ast.BMul None }  | "*."  { Ast.BMul (Some $1) }
  | "/" { Ast.BDiv None }  | "/."  { Ast.BDiv (Some $1) }
  | "**" { Ast.BPow None } | "**." { Ast.BPow (Some $1) }

%inline unop:
  | "not" { Ast.UNot }

%inline defname:
  | name { Ast.DName $1 }
  | unop { Ast.DUnOp $1 }
  | binop { Ast.DBinOp $1 }

%inline index: Int { $1 }

%inline path: separated_nonempty_llist("::", name) { $1 }

%inline receivers: separated_nonempty_llist(",", receiver) { $1 }
%inline receiver: pat0 "in" expr0 "=>" expr0 { ($1, $3, $5) }

%inline handler:
  | receiver { [$1] }
  | brace(receivers) { $1 }

expr0:
  | "on" handler { Ast.EOn $2 }
  | "return" expr1? { Ast.EReturn $2 }
  | "break" expr1? { Ast.EBreak $2 }
  | "continue" { Ast.EContinue }
  | expr1 { $1 }
  
expr1:
  | "fun" params annot(ty0)? body { Ast.EFunc ($2, $4) }
  | "task" ":" params body { Ast.ETask ($3, $4) }
  | expr2 { $1 }
  
%inline op2:
  | "=" { Ast.BMut }
  | "in" { Ast.BIn }
  | "not" "in" { Ast.BNotIn }
expr2:
  | expr3 { $1 }
  | expr2 "!" expr3 { Ast.EEmit ($1, $3) }
  | expr2 op2 expr3 { Ast.EBinOp ($2, $1, $3)}

%inline op3:
  | ".." { Ast.BRInc }
  | "..=" { Ast.BRExc }
expr3:
  | expr4 { $1 }
  | expr4 op3 expr4 { Ast.EBinOp ($2, $1, $3)}
  
%inline op4:
  | "bor" { Ast.BBor }
  | "band" { Ast.BBand }
  | "bxor" { Ast.BBxor }
  | "or" { Ast.BOr }
  | "xor" { Ast.BXor }
  | "and" { Ast.BAnd }
expr4:
  | expr5 { $1 }
  | expr4 op4 expr5 { Ast.EBinOp ($2, $1, $3)}
  
%inline op5:
  | "==" { Ast.BEq None } | "==." { Ast.BEq (Some $1) }
  | "!=" { Ast.BNeq None } | "!=." { Ast.BNeq (Some $1) }
expr5:
  | expr6 { $1 }
  | expr5 op5 expr6 { Ast.EBinOp ($2, $1, $3)}

%inline op6:
  | "<" { Ast.BLt None }   | "<." { Ast.BLt (Some $1) }
  | ">" { Ast.BGt None }   | ">." { Ast.BGt (Some $1) }
  | "<=" { Ast.BLeq None } | "<=." { Ast.BLeq (Some $1) }
  | ">=" { Ast.BGeq None } | ">=." { Ast.BGeq (Some $1) }
  
expr6:
  | expr7 { $1 }
  | expr6 op6 expr7 { Ast.EBinOp ($2, $1, $3)}

%inline op7:
  | "+" { Ast.BAdd None } | "+." { Ast.BAdd (Some $1) }
  | "-" { Ast.BSub None } | "-." { Ast.BSub (Some $1) }
  | "%" { Ast.BMod None } | "%." { Ast.BMod (Some $1) }
expr7:
  | expr8 { $1 }
  | expr7 op7 expr8 { Ast.EBinOp ($2, $1, $3)}
  
%inline op8:
  | "*" { Ast.BMul None } | "*." { Ast.BMul (Some $1) }
  | "/" { Ast.BDiv None } | "/." { Ast.BDiv (Some $1) }
expr8:
  | expr9 { $1 }
  | expr8 op8 expr9 { Ast.EBinOp ($2, $1, $3)}

%inline op9:
  | "not" { Ast.UNot }
  | "-" { Ast.UNeg None } | "-." { Ast.UNeg (Some $1) }
expr9:
  | expr10 { $1 }
  | op9 expr9 { Ast.EUnOp ($1, $2)}

%inline op10:
  | "**" { Ast.BPow None } | "**." { Ast.BPow (Some $1) }
expr10:
  | expr11 { $1 }
  | expr11 op10 expr10 { Ast.EBinOp ($2, $1, $3)}

expr11:
  | expr12 { $1 }
  | expr11 alias(ty1) { Ast.ECast ($1, $2) }

%inline expr12:
  | expr13 { $1 }
  | expr14 { $1 }

expr13:
  | expr15 { $1 }
  | expr15 paren(separated_llist_trailing(",", expr1))
    { Ast.ECall ($1, $2) }

expr14:
  | expr12 "." index
    { Ast.EProject ($1, $3) }
  | expr12 "." name
    { Ast.EAccess ($1, $3) }
  | expr12 brack(expr1)
    { Ast.ESelect ($1, $2) }
  | expr12 "." name paren(separated_llist_trailing(",", expr1))
    { Ast.EInvoke ($1, $3, $4) }

%inline expr15:
  | paren(expr0)
    { $1 }
  | paren(binop)
    { Ast.EBinOpRef ($1) }
  | block
    { Ast.EBlock $1 }
  | lit
    { Ast.ELit $1 }
  | path loption(qualify(brack(tys)))
    { Ast.EPath ($1, $2) }
  | "[" separated_llist_trailing(",", expr0) tail(expr0)? "]"
    { Ast.EArray ($2, $3) }
  | "[" expr0 ";" for_generator ";" separated_llist(";", clause) "]"
    { Ast.ECompr ($2, $4, $6) }
  | tuple(expr0)
    { Ast.ETuple $1 }
  | record(expr0)
    { Ast.ERecord $1 }
  | "if" expr2 block else_block?
    { Ast.EIf ($2, $3, $4) }
  | "if" "val" pat0 "=" expr1 block else_block?
    { Ast.EIfVal ($3, $5, $6, $7) }
  | "match" expr1 brace(arms)
    { Ast.EMatch ($2, $3) }
  | "loop" block
    { Ast.ELoop $2 }
  | "while" expr1 block
    { Ast.EWhile ($2, $3) }
  | "while" "val" pat0 "=" expr1 block
    { Ast.EWhileVal ($3, $5, $6) }
  | "for" pat0 "in" expr1 block
    { Ast.EFor ($2, $4, $5) }
  | "from" separated_nonempty_llist(",", scan) brace(nonempty_llist(step))
    { Ast.EFrom ($2, $3) }
  | "_"
    { Ast.EAnon }

%inline scan: pat0 scankind expr1 { ($1, $2, $3) }
%inline scankind:
  | "in" { Ast.ScIn }
  | "=" { Ast.ScEq }

%inline step:
  | "where" expr1
    { Ast.SWhere $2 }
  | "join" scan join_on?
    { Ast.SJoin ($2, $3) }
  | "group" separated_nonempty_llist(",", expr1) window? loption(reduce)
    { Ast.SGroup ($2, $3, $4) }
  | "order" separated_nonempty_llist(",", pair(expr1, order))
    { Ast.SOrder $2 }
  | "yield" expr1
    { Ast.SYield $2 }

%inline window: "window" window_step? window_duration { ($2, $3) }
%inline window_step: "step" expr1 { $2 }
%inline window_duration: "duration" expr1 { $2 }
%inline join_on: "on" expr1 { $2 }
%inline reduce: "reduce" separated_nonempty_llist(",", agg) { $2 }

%inline order:
  | { Ast.OAsc }
  | "desc" { Ast.ODesc }

%inline agg: expr1 aggof? { ($1, $2) }
%inline aggof: "of" expr1 { $2 }

%inline qualify(x): "::" x { $2 }
%inline tail(x): "|" x { $2 }

%inline clause:
  | for_generator { let (x0, x1) = $1 in Ast.CFor (x0, x1) }
  | guard { Ast.CIf ($1) }

%inline for_generator: "for" pat0 "in" expr0 { ($2, $4) }
%inline guard: "if" expr0 { $2 }

%inline else_block: "else" block { $2 }

%inline arms: separated_nonempty_llist_trailing(",", arm) { $1 }
%inline arm: pat0 "=>" expr0 { ($1, $3) }

%inline block: "{" stmts expr0? "}" { ($2, $3) }

%inline stmts: llist(stmt) { $1 }
%inline stmt:
  | ";"
    { Ast.SNoop }
  | expr0 ";"
    { Ast.SExpr $1 }
  | "val" param "=" expr0 ";"
    { Ast.SVal ($2, $4) }
  | "var" name annot(ty0)? "=" expr0 ";"
    { Ast.SVar (($2, $3), $5) }

%inline pats: separated_nonempty_llist(",", pat0) { $1 }
pat0:
  | pat0 "or" pat1
  | pat1 { $1 }
  
pat1:
  | lit 
    { Ast.PConst $1 }
  | name
    { Ast.PVar $1 }
  | path paren(pats)
    { Ast.PUnwrap ($1, $2) }
  | tuple(pat0)
    { Ast.PTuple $1 }
  | record(pat0)
    { Ast.PRecord $1 }
  | "_"
    { Ast.PIgnore }

%inline tys: separated_nonempty_llist(",", ty0) { $1 }
ty0:
  | ty1
    { $1 }
  | "fun" paren(tys) ":" ty0
    { Ast.TFunc ($2, $4) }

ty1:
  | path loption(brack(tys))
    { Ast.TPath ($1, $2) }
  | tuple(ty0)
    { Ast.TTuple $1 }
  | record(ty0)
    { Ast.TRecord $1 }
  | brack(ty0)
    { Ast.TArray $1 }

%inline tuple(x): "(" x "," separated_llist_trailing(",", x) ")"
  { $2::$4 }

%inline record(x): "#{" fields(x) tail(x)? "}"
  { ($2, $3) }

%inline fields(x): separated_llist_trailing(",", field(x)) { $1 }
%inline field(x):
  | name ":" x
    { ($1, Some $3) }
  | name
    { ($1, None) }

%inline lit:
  | Bool { Ast.LBool $1 }
  | Char { Ast.LChar $1 }
  | Int { Ast.LInt ($1, None) }
  | Float { Ast.LFloat ($1, None) }
  | IntSuffix { Ast.LInt (fst $1, Some (snd $1)) }
  | FloatSuffix { Ast.LFloat (fst $1, Some (snd $1)) }
  | "unit" { Ast.LUnit }
  | String { Ast.LString $1 }
