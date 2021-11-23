%token ParenL
%token ParenR
(* %token ParenLR *)
%token BrackL
%token BrackR
(* %token BrackLR *)
%token PercentBraceL
%token BraceL
%token BraceR
(* %token BraceLR *)
%token AngleL
%token AngleR
(* %token AngleLR *)
(*= Operators ==============================================================*)
%token Neq
%token Percent
%token Star
%token StarStar
%token Plus
%token Comma
%token Minus
%token ArrowR
%token Dot
%token DotDot
%token DotDotEq
%token Slash
%token Colon
%token ColonColon
%token Semi
%token Leq
%token Eq
%token EqEq
%token Imply
%token Geq
(* %token AtSign *)
%token Underscore
%token Bar
(* %token BarBar *)
%token Tilde
(* %token Ampersand *)
(*= Retired Operators =======================================================*)
(*     Pipe, *)
(*     Bang, *)
(*     Dollar, *)
(*     Amp, *)
(*     AmpAmp, *)
(*     SemiSemi, *)
(*     ArrowL, *)
(*     Qm, *)
(*     QmQmQm, *)
(*     Caret, *)
(*= Keywords ================================================================*)
%token After
%token And
%token As
%token Break
%token Band
%token Bor
%token Bxor
%token By
%token Class
%token Continue
%token Def
%token Else
%token Emit
%token Enum
%token Every
%token Extern
%token For
(* %token From *)
%token Fun
%token If
%token In
%token Instance
(* %token Is *)
(* %token Let *)
%token Loop
%token Match
%token Mod
%token Not
%token On
%token Or
%token Return
%token Task
%token Type
%token Val
(* %token With *)
(* %token Var *)
%token Use
(* %token Xor *)
(*= Reserved Keywords =======================================================*)
(*     Add, *)
(*     Box, *)
(*     Do, *)
(*     End, *)
(*     Of, *)
(*     Port, *)
(*     Pub, *)
(*     Reduce, *)
(*     Shutdown, *)
(*     Sink, *)
(*     Source, *)
(*     Startup, *)
(*     State, *)
(*     Then, *)
(*     Timeout, *)
(*     Timer, *)
(*     Trigger, *)
(*     Where, *)
(*= Identifiers and Literals ================================================*)
%token <string> Name
%token <int> Int 
%token <float> Float
%token <bool> Bool
%token <char> Char
%token Unit
%token <string> String
(* %token <int> DurationNs  *)
(* %token <int> DurationUs  *)
(* %token <int> DurationMs  *)
(* %token <int> DurationS  *)
(* %token <int> DurationM  *)
(* %token <int> DurationH  *)
(* %token <int> DurationD  *)
(* %token <int> DurationW  *)
(*     LitDurationMo *)
(*     LitDurationY *)
(* %token <string> Date *)
(* %token <string> DateTime *)
(* %token <string> DateTimeZone *)
%token Eof
%start <Ast.ast> program
%start <Ast.expr> expr
%%

expr: expr0 Eof { $1 }
program: llist(item) Eof { $1 }

%inline item:
  | Extern Def name loption(generics) basic_params ty_annot Semi
    { Ast.IExternDef ($3, $4, $5, $6) }
  | Type name Eq ty0 Semi
    { Ast.ITypeAlias ($2, $4) }
  | Extern Type name loption(generics) Semi
    { Ast.IExternType ($3, $4) }
  | Enum name loption(generics) brace(variants)
    { Ast.IEnum ($2, $3, $4) }
  | Class name loption(generics) brace(llist(decl))
    { Ast.IClass ($2, $3, $4) }
  | Instance loption(generics) path loption(ty_args) brace(llist(def))
    { Ast.IInstance ($2, $3, $4, $5) }
  | Def name loption(generics) params ioption(ty_annot) body
    { Ast.IDef ($2, $3, $4, $5, $6) }
  | Mod name brace(llist(item))
    { Ast.IMod ($2, $3) }
  | Task name loption(generics) params Colon intf ArrowR intf body
    { Ast.ITask ($2, $3, $4, $6, $8, $9) }
  | Use path ioption(alias) Semi
    { Ast.IUse ($2, $3) }
  | Val name ioption(ty_annot) Eq expr0 Semi
    { Ast.IVal ($2, $3, $5) }

%inline decl: Def name loption(generics) params ioption(ty_annot)
  { ($2, $3, $4, $5) }

%inline def: Def name loption(generics) params ioption(ty_annot) block
  { ($2, $3, $4, $5, $6) }

%inline body:
  | Semi { None }
  | block { Some $1 }

%inline ty_annot: Colon ty0 { $2 }

%inline basic_params: paren(separated_llist(Comma, basic_param)) { $1 }
%inline basic_param: name Colon ty0 { ($1, $3) }

%inline params: paren(separated_llist(Comma, param)) { $1 }
%inline param: pat0 ioption(ty_annot) { ($1, $2) }

generics: BrackL separated_llist(Comma, generic) BrackR { $2 }
generic: Name { $1 }
(* generic: Name separated_llist(And, bound) { ($1, $2) } *)
(* bound: Name { $1 } *)

intf:
  | paren(ports) { Ast.PTagged $1 }
  | ty0 { Ast.PSingle $1 }
ports: separated_llist(Comma, port) { $1 }
port: name paren(ty0) { ($1, $2) }

%inline variants: separated_llist(Comma, variant) { $1 }
%inline variant: name loption(paren(separated_nonempty_llist(Comma, ty0))) { ($1, $2) }

%inline alias: As name { $2 }
%inline name: Name { $1 }
%inline index: Int { $1 }
%inline path: separated_nonempty_llist(ColonColon, name) { $1 }

%inline paren(x): ParenL x ParenR { $2 }
%inline brack(x): BrackL x BrackR { $2 }
%inline brace(x): BraceL x BraceR { $2 }
%inline angle(x): AngleL x AngleR { $2 }

expr0:
  | On handler { Ast.EOn $2 }
  | Emit expr0 { Ast.EEmit $2 }
  | After expr1 block { Ast.EAfter ($2, $3) }
  | Every expr1 block { Ast.EEvery ($2, $3) }
  | Return ioption(expr0) { Ast.EReturn $2 }
  | Break ioption(expr0) { Ast.EBreak $2 }
  | Continue { Ast.EContinue }
  | expr1 { $1 }

%inline handler:
  | arm { [$1] }
  | brace(separated_llist(Comma, arm)) { $1 }
  
expr1:
  | expr2 { $1 }
  
%inline op2:
  | Eq { Ast.BMut }
  | In { Ast.BIn }
  | Not In { Ast.BNotIn }
expr2:
  | expr3 { $1 }
  | expr2 op2 expr3 { Ast.EBinOp ($2, $1, $3)}

%inline op3:
  | DotDot { Ast.BRInc }
  | DotDotEq { Ast.BRExc }
expr3:
  | expr4 { $1 }
  | expr4 op3 expr4 { Ast.EBinOp ($2, $1, $3)}
  
%inline op4:
  | By { Ast.BBy }
  | Bor { Ast.BBor }
  | Band { Ast.BBand }
  | Bxor { Ast.BBxor }
  | Or { Ast.BOr }
  | And { Ast.BAnd }
expr4:
  | expr5 { $1 }
  | expr4 op4 expr5 { Ast.EBinOp ($2, $1, $3)}
  
%inline op5:
  | EqEq { Ast.BEq }
  | Neq { Ast.BNeq }
expr5:
  | expr6 { $1 }
  | expr5 op5 expr6 { Ast.EBinOp ($2, $1, $3)}

%inline op6:
  | AngleR { Ast.BGt }
  | AngleL { Ast.BLt }
  | Geq { Ast.BGeq }
  | Leq { Ast.BLeq }
  
expr6:
  | expr7 { $1 }
  | expr6 op6 expr7 { Ast.EBinOp ($2, $1, $3)}

%inline op7:
  | Plus { Ast.BAdd }
  | Minus { Ast.BSub }
  | Percent { Ast.BMod }
expr7:
  | expr8 { $1 }
  | expr7 op7 expr8 { Ast.EBinOp ($2, $1, $3)}
  
%inline op8:
  | Star { Ast.BMul }
  | Slash { Ast.BDiv }
expr8:
  | expr9 { $1 }
  | expr8 op8 expr9 { Ast.EBinOp ($2, $1, $3)}

%inline op9:
  | Not { Ast.UNot }
  | Minus { Ast.UNeg }
expr9:
  | expr10 { $1 }
  | op9 expr9 { Ast.EUnOp ($1, $2)}
  | Fun params Colon expr9 { Ast.EFunc ($2, $4) }
  | Task Colon expr9 { Ast.ETask ($3) }

%inline op10:
  | StarStar { Ast.BPow }
expr10:
  | expr11 { $1 }
  | expr11 op10 expr10 { Ast.EBinOp ($2, $1, $3)}

expr11:
  | expr12 { $1 }
  | expr11 As ty1 { Ast.ECast ($1, $3) }

%inline expr12:
  | expr13 { $1 }
  | expr14 { $1 }

expr13:
  | expr15 { $1 }
  | expr15 paren(separated_llist(Comma, expr1))
    { Ast.ECall ($1, $2) }

expr14:
  | expr12 Dot index
    { Ast.EProject ($1, $3) }
  | expr12 Dot name
    { Ast.EAccess ($1, $3) }
  | expr12 brack(expr1)
    { Ast.ESelect ($1, $2) }
  | expr12 Dot name paren(separated_llist(Comma, expr1))
    { Ast.EInvoke ($1, $3, $4) }

%inline expr15:
  | paren(expr0)
    { $1 }
  | block
    { Ast.EBlock $1 }
  | lit
    { Ast.ELit $1 }
  | path loption(qualified_ty_args)
    { Ast.EPath ($1, $2) }
  | BrackL separated_llist(Comma, expr0) option(tail(expr0)) BrackR
    { Ast.EArray ($2, $3) }
  | BrackL expr0 Semi for_generator Semi separated_llist(Semi, clause) BrackR
    { Ast.ECompr ($2, $4, $6) }
  | ParenL expr0 Comma separated_llist(Comma, expr0) ParenR
    { Ast.ETuple ($2::$4) }
  | PercentBraceL separated_llist(Comma, field_expr(expr0)) option(tail(expr0)) BraceR
    { Ast.ERecord ($2, $3) }
  | If expr2 block ioption(else_block)
    { Ast.EIf ($2, $3, $4) }
  | If Val pat0 Eq expr1 block ioption(else_block)
    { Ast.EIfVal ($3, $5, $6, $7) }
  | Match expr1 brace(separated_nonempty_llist(Comma, arm))
    { Ast.EMatch ($2, $3) }
  | Loop block
    { Ast.ELoop $2 }
  | For pat0 In expr1 block
    { Ast.EFor ($2, $4, $5) }

%inline qualified_ty_args: ColonColon ty_args { $2 }
%inline ty_args: brack(separated_nonempty_llist(Comma, ty0)) { $1 }
%inline tail(x): Bar x { $2 }

%inline clause:
  | for_generator { let (x0, x1) = $1 in Ast.CFor (x0, x1) }
  | guard { Ast.CIf ($1) }

%inline for_generator: For pat0 In expr0 { ($2, $4) }
%inline guard: If expr0 { $2 }

%inline else_block: Else block { $2 }

%inline arm: pat0 Imply expr0 { ($1, $3) }

%inline field_expr(expr):
  | name Colon expr
    { ($1, Some $3) }
  | name
    { ($1, None) }

block: BraceL llist(stmt) ioption(expr0) BraceR { ($2, $3) }

%inline llist(x):
  llist_rev(x) { $1 |> List.rev }

llist_rev(x):
  | { [] }
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
  | { [] }
  | separated_nonempty_llist(s, x) { $1 }

%inline stmt:
  | Semi
    { Ast.SNoop }
  | expr0 Semi
    { Ast.SExpr $1 }
  | Val param Eq expr0 Semi
    { Ast.SVal ($2, $4) }

pat0:
  | pat0 Or pat1
  | pat1 { $1 }
  
pat1:
  | lit 
    { Ast.PConst $1 }
  | name
    { Ast.PVar $1 }
  | path paren(separated_nonempty_llist(Comma, pat0))
    { Ast.PUnwrap ($1, $2) }
  | ParenL pat0 Comma separated_llist(Comma, pat0) ParenR
    { Ast.PTuple ($2::$4) }
  | PercentBraceL separated_llist(Comma, field_pat) BraceR
    { Ast.PRecord ($2) }
  | Underscore
    { Ast.PIgnore }

%inline field_pat:
  | name Colon pat0
    { ($1, Some $3) }
  | name
    { ($1, None) }

ty0:
  | ty1
    { $1 }
  | Fun paren(separated_llist(Comma, ty0)) Colon ty0
    { Ast.TFunc ($2, $4) }

ty1:
  | path loption(ty_args)
    { Ast.TPath ($1, $2) }
  | ParenL ty0 Comma separated_llist(Comma, ty0) ParenR
    { Ast.TTuple ($2::$4) }
  | PercentBraceL separated_llist(Comma, field_ty) option(tail(ty0)) BraceR
    { Ast.TRecord ($2, $3) }
  | brack(ty0)
    { Ast.TArray $1 }

%inline field_ty: name Colon ty0 { ($1, $3) }

%inline lit:
  | Bool { Ast.LBool $1 }
  | Char { Ast.LChar $1 }
  | Int { Ast.LInt ($1, None) }
  | Float { Ast.LFloat ($1, None) }
  | Unit { Ast.LUnit }
  | String { Ast.LString $1 }
