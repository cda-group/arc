grammar Arc;

import ArcLexerRules;

// top level rules
program : macros body=expr;
macros : macro*;
expr : letExpr | lambdaExpr | operatorExpr;
type: TI8 # I8
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
	| TUnit # Unit
	| TVec '[' elemT=type ']' # Vec
	| TStream '[' elemT=type ']' # Stream
	| TSimd '[' elemT=type ']' # Simd
	| TAppender '[' elemT=type ']' # Appender
	| TStreamAppender '[' elemT=type ']' # StreamAppender
	| TMerger '[' elemT=type ',' opT=commutativeBinop ']' # Merger
	| TDict '[' keyT=type ',' valueT=type ']' # Dict
	| TDictMerger '[' keyT=type ',' valueT=type ',' opT=commutativeBinop ']' # DictMerger
	| TVecMerger '[' elemT=type ',' opT=commutativeBinop ']' # VecMerger
	| '{' types+=type (',' types+=type)* '}' # Struct
	| TBarBar '(' returnT=type ')' # UnitFunction
	| TBar paramTypes+=type (',' paramTypes+=type)* TBar '(' returnT=type ')' # ParamFunction
	;

// inner rules
macro : TMacro name=TIdentifier '(' macroParams ')' TEqual body=expr ';';
macroParams : names+=TIdentifier (',' names+=TIdentifier)*;

typeAnnot : ':' type;


letExpr : TLet name=TIdentifier typeAnnot? TEqual value=operatorExpr ';' body=expr;
lambdaExpr	: TBarBar body=expr # UnitLambda
			| TBar lambdaParams TBar body=expr # ParamLambda
			;
lambdaParams : params+=param (',' params+=param)*;
param : name=TIdentifier typeAnnot?;
operatorExpr	: literalExpr # Literal
				| TIdentifier # Ident
				;
literalExpr	: TI8Lit # I8Lit
			| TI16Lit # I17Lit
			| TI32Lit # I32Lit
			| TI64Lit # I64Lit
			| TF32Lit # F32Lit
			| TF64Lit # F64Lit
			| TBoolLit # BoolLit
			| TStringLit # StringLit
			; 

commutativeBinop	: TPlus # Sum
					| TStar # Product
					| TMax # Max
					| TMin # Min
					;