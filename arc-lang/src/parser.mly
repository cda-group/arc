%{
  open Error
%}

%start <Ast.ast> program
%start <Ast.expr> expr
%%

expr: expr0 Eof { $1 }
program: items Eof { $1 }

(* Utilities *)
paren(x): "(" x ")" { $2 }
brack(x): "[" x "]" { $2 }
brace(x): "{" x "}" { $2 }
angle(x): "<" x ">" { $2 }

%inline epsilon: {}

fst(a, b): a b { $1 }
snd(a, b): a b { $2 }

%inline llist(x):
  llist_rev(x) { $1 |> List.rev }

llist_rev(x):
  | epsilon { [] }
  | llist_rev(x) x { $2::$1 }

nonempty_llist(x):
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

separated_llist_trailing(s, x):
  | separated_llist(s, x) s? { $1 }

separated_nonempty_llist_trailing(s, x):
  | separated_nonempty_llist(s, x) s? { $1 }

seq(x): separated_llist_trailing(",", x) { $1 }

(* The grammar *)

decorator: loption(snd("@", brace(seq(field(lit))))) { $1 }

items: llist(item) { $1 }
item:
  | decorator "extern" boption("async") "def" def_name generics paren(seq(ty0)) annot(ty0)? where ";"
    { Ast.IExternDef (Loc $loc, $1, $3, $5, $6, $7, $8, $9) }
  | decorator boption("async") "def" def_name generics params annot(ty0)? where body
    { Ast.IDef (Loc $loc, $1, $2, $4, $5, $6, $7, $8, $9) }
  | decorator "task" def_name generics params ":" sinks where body
    { Ast.ITask (Loc $loc, $1, $3, $4, $5, $7, $8, $9) }
  | decorator "type" name generics "=" ty0 where ";"
    { Ast.IType (Loc $loc, $1, $3, $4, $6, $7) }
  | decorator "extern" "type" name generics where ";"
    { Ast.IExternType (Loc $loc, $1, $4, $5, $6) }
  | decorator "class" name generics where brace(llist(decl))
    { Ast.IClass (Loc $loc, $1, $3, $4, $5, $6) }
  | decorator "instance" generics path loption(brack(seq(ty0))) where brace(llist(def))
    { Ast.IInstance (Loc $loc, $1, $3, $4, $5, $6, $7) }
  | decorator "mod" name brace(items)
    { Ast.IMod (Loc $loc, $1, $3, $4) }
  | decorator "use" path use_suffix? ";"
    { Ast.IUse (Loc $loc, $1, $3, $4) }
  | decorator "val" name annot(ty0)? "=" expr0 ";"
    { Ast.IVal (Loc $loc, $1, $3, $4, $6) }

decl: "def" name generics params annot(ty0)? where ";"
  { ($2, $3, $4, $5, $6) }

def: "def" name generics params annot(ty0)? where body
  { ($2, $3, $4, $5, $6, $7) }

body:
  | "=" expr1 { ([], Some $2) }
  | block { $1 }

annot(x): ":" x { $2 }

sinks: paren(seq(sink)) { $1 }
sink: name annot(ty0)? { ($1, $2) }

params: paren(seq(param)) { $1 }
param: pat0 annot(ty0)? { ($1, $2) }

generics: loption(brack(seq(generic))) { $1 }
generic: name { $1 }

where: lopt(snd("where", seq(bound))) { $1 }
bound: path brack(seq(ty0)) { ($1, $2) }

alias(x): "as" x { $2 }

name: Name { $1 }

binop:
  | op3 { $1 }
  | op4 { $1 }
  | op5 { $1 }
  | op6 { $1 }
  | op7 { $1 }
  | op8 { $1 }
  | op10 { $1 }

unop:
  | "not" { Ast.UNot }
  | "neg" { Ast.UNeg }

def_name:
  | name { Ast.DName $1 }
  | unop lopt(qualify(seq(ty0))) { Ast.DUnOp ($1, $2) }
  | binop lopt(qualify(seq(ty0))) { Ast.DBinOp ($1, $2) }

index: Int { $1 }

%inline path:
  | separated_nonempty_llist("::", name) { Ast.PRel $1 }
  | "::" separated_nonempty_llist("::", name) { Ast.PAbs $2 }
use_suffix:
  | "*" { Ast.UGlob }
  | alias(name) { Ast.UAlias $1 }

receivers: separated_nonempty_llist_trailing(",", receiver) { $1 }
receiver: pat0 "in" expr0 "=>" expr0 { ($1, $3, $5) }

handler: brace(receivers) { $1 }

expr0:
  | "on" handler { Ast.EOn (Loc $loc, $2) }
  | "return" expr1? { Ast.EReturn (Loc $loc, $2) }
  | "break" expr1? { Ast.EBreak (Loc $loc, $2) }
  | "continue" { Ast.EContinue (Loc $loc) }
  | "throw" expr1 { Ast.EThrow (Loc $loc, $2) }
  | expr1 { $1 }
  
expr1:
  | "fun" params annot(ty0)? body { Ast.EFunc (Loc $loc, $2, $4) }
  | "task" params ":" sinks body { Ast.ETask (Loc $loc, $2, $4, $5) }
  | expr2 { $1 }
  
op2:
  | "=" { Ast.BMut }
  | "in" { Ast.BIn }
  | "not" "in" { Ast.BNotIn }
expr2:
  | expr3 { $1 }
  | expr2 op2 expr3 { Ast.EBinOp (Loc $loc, $2, [], $1, $3)}

op3:
  | ".." { Ast.BRExc }
  | "..=" { Ast.BRInc }
expr3:
  | expr4 { $1 }
  | expr4 op3 lopt(qualify(seq(ty0))) expr4 { Ast.EBinOp (Loc $loc, $2, $3, $1, $4)}
  
op4:
  | "bor" { Ast.BBor }
  | "band" { Ast.BBand }
  | "bxor" { Ast.BBxor }
  | "or" { Ast.BOr }
  | "xor" { Ast.BXor }
  | "and" { Ast.BAnd }
expr4:
  | expr5 { $1 }
  | expr4 op4 lopt(qualify(seq(ty0))) expr5 { Ast.EBinOp (Loc $loc, $2, $3, $1, $4)}

op5:
  | "==" { Ast.BEq }
  | "!=" { Ast.BNeq }
expr5:
  | expr6 { $1 }
  | expr5 op5 lopt(qualify(seq(ty0))) expr6 { Ast.EBinOp (Loc $loc, $2, $3, $1, $4)}

op6:
  | "<" { Ast.BLt }
  | ">" { Ast.BGt }
  | "<=" { Ast.BLeq }
  | ">=" { Ast.BGeq }
expr6:
  | expr7 { $1 }
  | expr6 op6 lopt(qualify(seq(ty0))) expr7 { Ast.EBinOp (Loc $loc, $2, $3, $1, $4)}

op7:
  | "+" { Ast.BAdd }
  | "-" { Ast.BSub }
  | "%" { Ast.BMod }
expr7:
  | expr8 { $1 }
  | expr7 op7 lopt(qualify(seq(ty0))) expr8 { Ast.EBinOp (Loc $loc, $2, $3, $1, $4)}
  
op8:
  | "*" { Ast.BMul }
  | "/" { Ast.BDiv }
expr8:
  | expr9 { $1 }
  | expr8 op8 lopt(qualify(seq(ty0))) expr9 { Ast.EBinOp (Loc $loc, $2, $3, $1, $4)}

op9:
  | "not" { Ast.UNot }
  | "-" { Ast.UNeg }
expr9:
  | expr10 { $1 }
  | op9 lopt(qualify(seq(ty0))) expr9 { Ast.EUnOp (Loc $loc, $1, $2, $3)}

op10:
  | "**" { Ast.BPow }
expr10:
  | expr11 { $1 }
  | expr11 op10 lopt(qualify(seq(ty0))) expr10 { Ast.EBinOp (Loc $loc, $2, $3, $1, $4)}

expr11:
  | expr12 { $1 }
  | expr11 alias(ty1) { Ast.ECast (Loc $loc, $1, $2) }

expr12:
  | expr13 { $1 }
  | expr14 { $1 }

expr13:
  | expr15 { $1 }
  | expr15 paren(seq(expr1))
    { Ast.ECall (Loc $loc, $1, $2) }

expr14:
  | expr12 "." index
    { Ast.EProject (Loc $loc, $1, $3) }
  | expr12 "." name
    { Ast.EAccess (Loc $loc, $1, $3) }
  | expr12 brack(expr1)
    { Ast.ESelect (Loc $loc, $1, $2) }
  | expr12 "." name paren(seq(expr1))
    { Ast.EInvoke (Loc $loc, $1, $3, $4) }

%inline lopt(l): ioption(l) { match $1 with None -> [] | Some l -> l }

expr15:
  | paren(expr0)
    { $1 }
  | paren(binop)
    { Ast.EBinOpRef (Loc $loc, $1) }
  | block
    { Ast.EBlock (Loc $loc, $1) }
  | lit
    { Ast.ELit (Loc $loc, $1) }
  | path loption(qualify(seq(ty0)))
    { Ast.EPath (Loc $loc, $1, $2) }
  | array(expr0)
    { Ast.EArray (Loc $loc, fst $1, snd $1) }
  | tuple(expr0)
    { Ast.ETuple (Loc $loc, $1) }
  | record(expr0)
    { Ast.ERecord (Loc $loc, $1) }
  | "if" expr2 block snd("else", block)?
    { Ast.EIf (Loc $loc, $2, $3, $4) }
  | "if" "val" pat0 "=" expr1 block snd("else", block)?
    { Ast.EIfVal (Loc $loc, $3, $5, $6, $7) }
  | "match" expr1 brace(arms)
    { Ast.EMatch (Loc $loc, $2, $3) }
  | "loop" block
    { Ast.ELoop (Loc $loc, $2) }
  | "while" expr1 block
    { Ast.EWhile (Loc $loc, $2, $3) }
  | "while" "val" pat0 "=" expr1 block
    { Ast.EWhileVal (Loc $loc, $3, $5, $6) }
  | "for" pat0 "in" expr1 block
    { Ast.EFor (Loc $loc, $2, $4, $5) }
  | "try" block "catch" brace(arms) snd("finally", block)?
    { Ast.ETry (Loc $loc, $2, $4, $5) }
  | "from" sources brace(clauses)
    { Ast.EFrom (Loc $loc, $2, $3) }
  | name record(expr0)
    { Ast.EEnwrap (Loc $loc, $1, ERecord (Loc $loc, $2)) }
  | name tuple(expr0)
    { Ast.EEnwrap (Loc $loc, $1, ETuple (Loc $loc, $2)) }
  | "_"
    { Ast.EAnon (Loc $loc) }

sources: separated_nonempty_llist(",", source) { $1 }
source: pat0 source_kind expr1 { ($1, $2, $3) }
source_kind:
  | "in" { Ast.ScIn (Loc $loc) }
  | "=" { Ast.ScEq (Loc $loc) }

clauses: nonempty_llist(clause) { $1 }
clause:
  | "where" expr1
    { Ast.SWhere (Loc $loc, $2) }
  | "join" source snd("on", expr1)?
    { Ast.SJoin (Loc $loc, $2, $3) }
  | "group" separated_nonempty_llist(",", expr1) window? loption(compute)
    { Ast.SGroup (Loc $loc, $2, $3, $4) }
  | "order" separated_nonempty_llist(",", pair(expr1, order))
    { Ast.SOrder (Loc $loc, $2) }
  | "yield" expr1
    { Ast.SYield (Loc $loc, $2) }

window: "window" snd("step", expr1)? snd("duration", expr1) { ($2, $3) }
compute: "compute" separated_nonempty_llist(",", pair(expr1, snd("of", expr1)?)) { $2 }

order:
  | epsilon { Ast.OAsc }
  | "desc" { Ast.ODesc }

qualify(x): "::" brack(x) { $2 }
tail(x): "|" x { $2 }

arms: separated_nonempty_llist_trailing(",", arm) { $1 }
arm: pat0 "=>" expr0 { ($1, $3) }

block: "{" stmts expr0? "}" { ($2, $3) }

%inline stmts: llist(stmt) { $1 }
stmt:
  | ";"
    { Ast.SNoop (Loc $loc) }
  | expr0 ";"
    { Ast.SExpr (Loc $loc, $1) }
  | "val" param "=" expr0 ";"
    { Ast.SVal (Loc $loc, $2, $4) }
  | "var" name annot(ty0)? "=" expr0 ";"
    { Ast.SVar (Loc $loc, ($2, $3), $5) }

pat0:
  | pat0 "or" pat1
  | pat1 { $1 }
  
pat1:
  | lit 
    { Ast.PConst (Loc $loc, $1) }
  | name
    { Ast.PVar (Loc $loc, $1) }
  | name record(pat0)
    { Ast.PUnwrap (Loc $loc, $1, PRecord (Loc $loc, $2)) }
  | name tuple(pat0)
    { Ast.PUnwrap (Loc $loc, $1, PTuple (Loc $loc, $2)) }
  | tuple(pat0)
    { Ast.PTuple (Loc $loc, $1) }
  | record(pat0)
    { Ast.PRecord (Loc $loc, $1) }
  | array(pat0)
    { Ast.PArray (Loc $loc, fst $1, snd $1) }
  | "_"
    { Ast.PIgnore (Loc $loc) }

ty0:
  | ty1
    { $1 }
  | "fun" paren(seq(ty0)) ":" ty0
    { Ast.TFunc (Loc $loc, $2, $4) }

ty1:
  | path loption(brack(separated_nonempty_llist(",", ty0)))
    { Ast.TPath (Loc $loc, $1, $2) }
  | tuple(ty0)
    { Ast.TTuple (Loc $loc, $1) }
  | record(ty0)
    { Ast.TRecord (Loc $loc, $1) }
  | enum(ty0)
    { Ast.TEnum (Loc $loc, $1) }
  | brack(ty0)
    { Ast.TArray (Loc $loc, $1) }

array(x): "[" seq(x) tail(x)? "]" { ($2, $3) }

tuple(x): "#(" seq(x) ")" { $2 }

record(x): "#{" seq(field(x)) tail(x)? "}" { ($2, $3) }
field(x): name snd(":", x)? { ($1, $2) }

enum(x): "enum" "{" seq(variant(x)) tail(x)? "}" { ($3, $4) }
variant(x):
  | name record(x) { ($1, Ast.TRecord (Loc $loc, $2)) }
  | name tuple(x) { ($1, Ast.TTuple (Loc $loc, $2))}

lit:
  | Bool { Ast.LBool (Loc $loc, $1) }
  | Char { Ast.LChar (Loc $loc, $1) }
  | Int { Ast.LInt (Loc $loc, $1, None) }
  | Float { Ast.LFloat (Loc $loc, $1, None) }
  | IntSuffix { Ast.LInt (Loc $loc, fst $1, Some (snd $1)) }
  | FloatSuffix { Ast.LFloat (Loc $loc, fst $1, Some (snd $1)) }
  | "unit" { Ast.LUnit (Loc $loc) }
  | String { Ast.LString (Loc $loc, $1) }
