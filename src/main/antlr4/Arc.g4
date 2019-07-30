grammar Arc;

import ArcLexerRules;

// top level rules
program: macros body=expr;
macros: macro*;

valueExpr
: letExpr
| operatorExpr
;

expr
: lambdaExpr
| valueExpr
;

type
: TI8 # I8
| TI16 # I16
| TI32 # I32
| TI64 # I64
| TU8 # U8
| TU16 # U16
| TU32 # U32
| TU64 # U64
| TF32 # F32
| TF64 # F64
| TBool # Bool
| TUnit # UnitT
| TString # StringT
| TVec '[' elemT=type ']' # Vec
| TStream '[' elemT=type ']' # Stream
| TSimd '[' elemT=type ']' # Simd
| annotations? TAppender '[' elemT=type ']' # Appender
| annotations? TStreamAppender '[' elemT=type ']' # StreamAppender
| annotations? TMerger '[' elemT=type ',' opT=commutativeBinop ']' # Merger
| TDict '[' keyT=type ',' valueT=type ']' # Dict
| annotations? TDictMerger '[' keyT=type ',' valueT=type ',' opT=commutativeBinop ']' # DictMerger
| annotations? TGroupMerger '[' keyT=type ',' valueT=type ']' # GroupMerger
| annotations? TVecMerger '[' elemT=type ',' opT=commutativeBinop ']' # VecMerger
| annotations? TWindower '[' discT=type ',' aggrT=type ',' aggrMergeT=type ',' aggrResultT=type ']' # Windower
| '{' types+=type (',' types+=type)* '}' # Struct
| TBarBar '(' returnT=type ')' # UnitFunction
| TBar paramTypes+=type (',' paramTypes+=type)* TBar '(' returnT=type ')' # ParamFunction
| TTypeVar # TypeVariable
;

// inner rules
macro: TMacro name=TIdentifier '(' macroParams ')' TEqual body=expr ';';
macroParams: names+=TIdentifier (',' names+=TIdentifier)*;

typeAnnot: ':' type;

letExpr: TLet name=TIdentifier typeAnnot? TEqual value=operatorExpr ';' body=valueExpr;

lambdaExpr
: TBarBar body=valueExpr # UnitLambda
| TBar lambdaParams TBar body=valueExpr # ParamLambda
;

lambdaParams: params+=param (',' params+=param)*;

param: name=TIdentifier typeAnnot?;

operatorExpr
: literalExpr # Literal
| scalar=(TI8|TI16|TI32|TI64|TU8|TU16|TU32|TU64|TF32|TF64|TBool) '(' valueExpr ')' # Cast
| TToVec '(' valueExpr ')' # ToVec
| TIdentifier # Ident
| '(' expr ')' # ParenExpr
| '[' entries+=valueExpr (',' entries+=valueExpr)* ']' # MakeVec
| '{' entries+=valueExpr (',' entries+=valueExpr)* '}' # MakeStruct
| annotations? TIf '(' cond=valueExpr ',' onTrue=valueExpr ',' onFalse=valueExpr ')' # If
| TIterate '(' initial=valueExpr ',' updateFunc=lambdaExpr ')' # Iterate
| TSelect '(' cond=valueExpr ',' onTrue=valueExpr ',' onFalse=valueExpr ')' # Select
| TBroadcast '(' valueExpr ')' # Broadcast
| TSerialize '(' valueExpr ')' # Serialize
| annotations? TDeserialize '[' type ']' '(' valueExpr ')' # Deserialize
| annotations? cudfExpr # CUDF
| TZip '(' functionParams ')' # Zip
| THash '(' functionParams ')' # Hash
| annotations? TFor '(' iterator ',' builder=valueExpr ',' body=lambdaExpr ')' # For
| TLen '(' valueExpr ')' # Len
| TLookup '(' data=valueExpr ',' key=valueExpr ')' # Lookup
| TSlice '(' data=valueExpr ',' index=valueExpr ',' size=valueExpr ')' # Slice
| TSort '(' data=valueExpr ',' keyFunc=lambdaExpr ')' # Sort
| TDrain '(' source=valueExpr ',' sink=valueExpr ')' # Drain
| unaryExpr # Unary
| TMerge '(' builder=valueExpr ',' value=valueExpr ')' # Merge
| TResult '(' valueExpr ')' # Result
| annotations? TAppender ('[' elemT=type ']')? ('(' arg=valueExpr ')')? # NewAppender
| annotations? TStreamAppender ('[' elemT=type ']')? ('(' ')')? # NewStreamAppender
| annotations? TMerger '[' elemT=type ',' commutativeBinop ']' ('(' arg=valueExpr ')')? # NewMerger
| annotations? TDictMerger '[' keyT=type ',' valueT=type ',' opT=commutativeBinop ']' ('(' arg=valueExpr ')')? # NewDictMerger
| annotations? TGroupMerger ('[' keyT=type ',' valueT=type ']')? ('(' arg=valueExpr ')')? # NewGroupMerger
| annotations? TVecMerger '[' elemT=type ',' opT=commutativeBinop ']' ('(' arg=valueExpr ')')? # NewVecMerger
| annotations? TWindower '[' discT=type ',' aggrT=type ',' aggrMergeT=type ',' aggrResultT=type ']' '(' assign=lambdaExpr ',' trigger=lambdaExpr ',' lower=lambdaExpr ')' # NewWindower
| fun=(TMin|TMax|TPow) '(' left=valueExpr ',' right=valueExpr ')' # BinaryFunction
| operatorExpr '(' functionParams ')' # Application
| operatorExpr '.' TIndex # Projection
| operatorExpr ':' type # Ascription
| left=operatorExpr op=(TStar|TSlash|TPercent) right=operatorExpr # Product
| left=operatorExpr op=(TPlus|TMinus) right=operatorExpr # Sum
| left=operatorExpr op=(TLessThan|TGreaterThan|TLEq|TGEq) right=operatorExpr # Comparison
| left=operatorExpr op=(TEqualEqual|TNotEqual) right=operatorExpr # Equality
| left=operatorExpr TAnd right=operatorExpr # BitwiseAnd
| left=operatorExpr TCirc right=operatorExpr # BitwiseXor
| left=operatorExpr TBar right=operatorExpr # BitwiseOr
| left=operatorExpr TAndAnd right=operatorExpr # LogicalAnd
| left=operatorExpr TBarBar right=operatorExpr # LogicalOr
;

unaryExpr
: TMinus operatorExpr # Negate
| TBang operatorExpr # Not
| op=(TExp | TSin | TCos | TTan | TASin | TACos | TATan | TSinh | TCosh | TTanh | TLog | TErf | TSqrt) '(' valueExpr ')' # UnaryOp
;

literalExpr
: TI8Lit # I8Lit
| TI16Lit # I16Lit
| TI32Lit # I32Lit
| TI64Lit # I64Lit
| TF32Lit # F32Lit
| TF64Lit # F64Lit
| TBoolLit # BoolLit
| TStringLit # StringLit
| TUnitLit # UnitLit
;

iterator
: iter=(TScalarIter|TSimdIter|TFringeIter|TNdIter) '(' data=valueExpr ')' # SimpleIter
| iter=(TScalarIter|TSimdIter|TFringeIter) '(' data=valueExpr ',' start=valueExpr ',' end=valueExpr ',' stride=valueExpr ')' # FourIter
| TNdIter '(' data=valueExpr ',' start=valueExpr ',' end=valueExpr ',' stride=valueExpr ',' shape=valueExpr ',' strides=valueExpr ')' # SixIter
| TRangeIter '(' start=valueExpr ',' end=valueExpr ',' stride=valueExpr ')' # RangeIter
| TKeyByIter '(' data=valueExpr ',' keyFunc=lambdaExpr ')' # KeyByIter
| valueExpr # UnkownIter
;

cudfExpr
: TCUDF '[' TStar funcPointer=valueExpr ',' returnType=type ']' '(' functionParams ')' # PointerUDF
| TCUDF '[' name=TIdentifier ',' returnType=type ']' '(' functionParams ')' # NameUDF
;

functionParams: params+=expr (',' params+=expr)*;

commutativeBinop
: TPlus # SumOp
| TStar # ProductOp
| TMax # MaxOp
| TMin # MinOp
;

annotations: TAt '(' entries+=annotationPair (',' entries+=annotationPair)* ')';

annotationPair
: name=TIdentifier ':' value=TIdentifier # IdPair
| name=TIdentifier ':' value=literalExpr # LiteralPair
;
