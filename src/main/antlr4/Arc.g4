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
	| annotations? TAppender '[' elemT=type ']' # Appender
	| annotations? TStreamAppender '[' elemT=type ']' # StreamAppender
	| annotations? TMerger '[' elemT=type ',' opT=commutativeBinop ']' # Merger
	| TDict '[' keyT=type ',' valueT=type ']' # Dict
	| annotations? TDictMerger '[' keyT=type ',' valueT=type ',' opT=commutativeBinop ']' # DictMerger
	| annotations? TGroupMerger '[' keyT=type ',' valueT=type ']' # GroupMerger
	| annotations? TVecMerger '[' elemT=type ',' opT=commutativeBinop ']' # VecMerger
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
				| scalar=(TI8|TI16|TI32|TI64|TU8|TU16|TU32|TU64|TF32|TF64|TBool) '(' expr ')' # Cast
				| TToVec '(' expr ')' # ToVec
				| TIdentifier # Ident
				| '(' expr ')' # ParenExpr
				| '[' entries+=expr (',' entries+=expr)* ']' # MakeVec
				| '{' entries+=expr (',' entries+=expr)* '}' # MakeStruct
				| annotations? TIf '(' cond=expr ',' onTrue=expr ',' onFalse=expr ')' # If
				| TIterate '(' initial=expr ',' updateFunc=expr ')' # Iterate
				| TSelect '(' cond=expr ',' onTrue=expr ',' onFalse=expr ')' # Select
				| TBroadcast '(' expr ')' # Broadcast
				| TSerialize '(' expr ')' # Serialize
				| annotations? TDeserialize '[' type ']' '(' expr ')' # Deserialize
				| annotations? cudfExpr # CUDF
				| TZip '(' functionParams ')' # Zip
				| annotations? TFor '(' iterator ',' builder=expr ',' body=expr ')' # For
				| TLen '(' expr ')' # Len
				| TLookup '(' data=expr ',' key=expr ')' # Lookup
				| TSlice '(' data=expr ',' index=expr ',' size=expr ')' # Slice
				| TSort '(' data=expr ',' keyFunc=expr ')' # Sort
				| unaryExpr # Unary
				| TMerge '(' builder=expr ',' value=expr ')' # Merge
				| TResult '(' expr ')' # Result
				| annotations? TAppender '[' elemT=type ']' ('(' arg=expr ')')? # NewAppender
				| annotations? TStreamAppender '[' elemT=type ']' ('(' ')')? # NewStreamAppender
				| annotations? TMerger '[' elemT=type ',' commutativeBinop ']' ('(' arg=expr ')')? # NewMerger
				| annotations? TDictMerger '[' keyT=type ',' valueT=type ',' opT=commutativeBinop ']' ('(' arg=expr ')')? # NewDictMerger
				| annotations? TGroupMerger '[' keyT=type ',' valueT=type ']' ('(' arg=expr ')')? # NewGroupMerger
				| annotations? TVecMerger '[' elemT=type ',' opT=commutativeBinop ']' ('(' arg=expr ')')? # NewVecMerger
				| fun=(TMin|TMax|TPow) '(' left=expr ',' right=expr ')' # BinaryFunction
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

unaryExpr	: TMinus expr # Negate
			| TBang expr # Not
			| op=(TExp | TSin | TCos | TTan | TASin | TACos | TATan | TSinh | TCosh | TTanh | TLog | TErf | TSqrt) '(' expr ')' # UnaryOp
			; 

literalExpr	: TI8Lit # I8Lit
			| TI16Lit # I16Lit
			| TI32Lit # I32Lit
			| TI64Lit # I64Lit
			| TF32Lit # F32Lit
			| TF64Lit # F64Lit
			| TBoolLit # BoolLit
			| TStringLit # StringLit
			; 

iterator	: iter=(TScalarIter|TSimdIter|TFringeIter|TNdIter) '(' data=expr ')' # SimpleIter
			| iter=(TScalarIter|TSimdIter|TFringeIter) '(' data=expr ',' start=expr ',' end=expr ',' stride=expr ')' # FourIter
			| TNdIter '(' data=expr ',' start=expr ',' end=expr ',' stride=expr ',' shape=expr ',' strides=expr ')' # SixIter
			| TRangeIter '(' start=expr ',' end=expr ',' stride=expr ')' # RangeIter
			| expr # UnkownIter
			;

cudfExpr	: TCUDF '[' TStar funcPointer=expr ',' returnType=type ']' '(' functionParams ')' # PointerUDF
			| TCUDF '[' name=TIdentifier ',' returnType=type ']' '(' functionParams ')' # NameUDF
			;

functionParams : params+=expr (',' params+=expr)*;

commutativeBinop	: TPlus # SumOp
					| TStar # ProductOp
					| TMax # MaxOp
					| TMin # MinOp
					;

annotations : TAt '(' entries+=annotationPair (',' entries+=annotationPair)* ')';
annotationPair	: name=TIdentifier ':' value=TIdentifier # IdPair
				| name=TIdentifier ':' value=literalExpr # LiteralPair
				;