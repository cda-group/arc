package se.kth.cda.arc

import org.antlr.v4.runtime._
import org.antlr.v4.runtime.tree._
import scala.util.matching.Regex
import scala.util.Try;
import collection.JavaConverters._

class ASTTranslator(val parser: ArcParser) {
  import AST._;
  import ASTTranslator._;
  import ArcParser._;

  def program(): Program = {
    val tree = parser.program();
    this.translate(tree)
  }
  def translate(tree: ArcParser.ProgramContext): Program = ProgramVisitor.visitChecked(tree);

  def expr(): Expr = {
    val tree = parser.expr();
    this.translate(tree)
  }
  def translate(tree: ArcParser.ExprContext): Expr = ExprVisitor.visitChecked(tree);

  def macros(): List[Macro] = {
    val tree = parser.macros();
    this.translate(tree)
  }
  def translate(tree: ArcParser.MacrosContext): List[Macro] = MacrosVisitor.visitChecked(tree);

  def `type`(): Type = {
    val tree = parser.`type`();
    this.translate(tree)
  }
  def translate(tree: ArcParser.TypeContext): Type = TypeVisitor.visitChecked(tree);

  private def tokenToSymbol(t: Token): Symbol = Symbol(t.getText, Some(t));
  private def annotToType(ctx: TypeAnnotContext): Option[Type] = {
    val annotO = Option(ctx);
    annotO.map(a => TypeVisitor.visitChecked(a));
  }
  private def annotToTypeBound(ctx: TypeAnnotContext): Type = {
    annotToType(ctx) match {
      case Some(ty) => ty
      case None     => Types.unknown
    }
  }

  object ProgramVisitor extends ArcBaseVisitor[Program] {
    def visitChecked(tree: ParseTree): Program = {
      val e = this.visit(tree);
      assert(e != null, s"Visiting a sub-tree returned null:\n${tree.toStringTree(parser)}");
      e
    }
    override def visitProgram(ctx: ProgramContext): Program = {
      val macros = MacrosVisitor.visitMacros(ctx.macros());
      assert(macros != null);
      val expr = ExprVisitor.visitExpr(ctx.expr());
      assert(expr != null);
      Program(macros, expr, ctx)
    }
  }
  object MacrosVisitor extends ArcBaseVisitor[List[Macro]] {
    def visitChecked(tree: ParseTree): List[Macro] = {
      val e = this.visit(tree);
      assert(e != null, s"Visiting a sub-tree returned null:\n${tree.toStringTree(parser)}");
      e
    }
    override def visitMacros(ctx: MacrosContext): List[Macro] = {
      val macros = ctx.`macro`().asScala.map(visitChecked(_)).flatten;
      macros.toList
    }
    override def visitMacro(ctx: MacroContext): List[Macro] = {
      val name = tokenToSymbol(ctx.name);
      val params = ctx.macroParams().names.asScala.map(tokenToSymbol(_));
      val body = ExprVisitor.visitChecked(ctx.body);
      List(Macro(name, params.toVector, body, ctx))
    }
  }

  object ExprVisitor extends ArcBaseVisitor[Expr] {
    def visitChecked(tree: ParseTree): Expr = {
      val e = this.visit(tree);
      assert(e != null, s"Visiting a sub-tree returned null:\n${tree.toStringTree(parser)}");
      e
    }

    override def visitParenExpr(ctx: ParenExprContext): Expr = {
      this.visitChecked(ctx.valueExpr())
    }

    override def visitLetExpr(ctx: LetExprContext): Expr = {
      val name = tokenToSymbol(ctx.name);
      assert(name != null);
      val tyO = annotToType(ctx.typeAnnot())
      val value = this.visitChecked(ctx.value);
      val ty = tyO match {
        case Some(ty) => ty
        case None     => Types.unknown
      }
      val body = this.visitChecked(ctx.body);
      Expr(ExprKind.Let(name, ty, value, body), body.ty, ctx)
    }
    override def visitUnitLambda(ctx: UnitLambdaContext): Expr = {
      val body = this.visitChecked(ctx.body);
      Expr(ExprKind.Lambda(Vector.empty, body), Types.unknown, ctx)
    }
    override def visitParamLambda(ctx: ParamLambdaContext): Expr = {
      val params = ctx.lambdaParams().param().asScala.map { p =>
        val name = tokenToSymbol(p.name);
        val ty = annotToTypeBound(p.typeAnnot());
        Parameter(name, ty)
      }.toVector;
      val body = this.visitChecked(ctx.body);
      Expr(ExprKind.Lambda(params, body), Types.unknown, ctx)
    }
    override def visitIdent(ctx: IdentContext): Expr = {
      Expr(ExprKind.Ident(tokenToSymbol(ctx.TIdentifier().getSymbol)), Types.unknown, ctx)
    }

    override def visitCast(ctx: ArcParser.CastContext): Expr = {
      val Some(ty) = TypeVisitor.tokenToScalar(ctx.scalar);
      val e = this.visitChecked(ctx.valueExpr());
      Expr(ExprKind.Cast(ty, e), ty, ctx)
    }

    override def visitToVec(ctx: ArcParser.ToVecContext): Expr = {
      val e = this.visitChecked(ctx.valueExpr());
      Expr(ExprKind.ToVec(e), Types.Vec(Types.unknown), ctx)
    }

    override def visitZip(ctx: ArcParser.ZipContext): Expr = {
      val params = ctx.functionParams().params.asScala.map(this.visitChecked(_)).toVector;
      Expr(ExprKind.Zip(params), Types.unknown, ctx)
    }

    override def visitFor(ctx: ArcParser.ForContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val iter = IterVisitor.visitChecked(ctx.iterator());
      val builder = this.visitChecked(ctx.builder);
      val body = this.visitChecked(ctx.body);
      Expr(ExprKind.For(iter, builder, body), builder.ty, ctx, annot)
    }

    override def visitLen(ctx: ArcParser.LenContext): Expr = {
      val e = this.visitChecked(ctx.valueExpr());
      Expr(ExprKind.Len(e), Types.I64, ctx)
    }

    override def visitLookup(ctx: ArcParser.LookupContext): Expr = {
      val key = this.visitChecked(ctx.key);
      val data = this.visitChecked(ctx.data);
      Expr(ExprKind.Lookup(data, key), Types.unknown, ctx)
    }

    override def visitSlice(ctx: ArcParser.SliceContext): Expr = {
      val data = this.visitChecked(ctx.data);
      val index = this.visitChecked(ctx.index);
      val size = this.visitChecked(ctx.size);
      Expr(ExprKind.Slice(data, index, size), data.ty, ctx)
    }

    override def visitSort(ctx: ArcParser.SortContext): Expr = {
      val data = this.visitChecked(ctx.data);
      val keyFunc = this.visitChecked(ctx.keyFunc);
      Expr(ExprKind.Sort(data, keyFunc), data.ty, ctx)
    }

    override def visitNegate(ctx: NegateContext): Expr = {
      val expr = this.visitChecked(ctx.operatorExpr());
      Expr(ExprKind.Negate(expr), expr.ty, ctx)
    }

    override def visitNot(ctx: NotContext): Expr = {
      val expr = this.visitChecked(ctx.operatorExpr());
      Expr(ExprKind.Not(expr), Types.Bool, ctx)
    }

    override def visitUnaryOp(ctx: UnaryOpContext): Expr = {
      import ArcLexer._;
      import UnaryOpKind._;
      val expr = this.visitChecked(ctx.valueExpr());
      val kind = ctx.op.getType match {
        case TExp  => Exp
        case TSin  => Sin
        case TCos  => Cos
        case TTan  => Tan
        case TASin => ASin
        case TACos => ACos
        case TATan => ATan
        case TSinh => Sinh
        case TCosh => Cosh
        case TTanh => Tanh
        case TLog  => Log
        case TErf  => Erf
        case TSqrt => Sqrt
      };
      Expr(ExprKind.UnaryOp(kind, expr), expr.ty, ctx)
    }

    override def visitMakeVec(ctx: MakeVecContext): Expr = {
      val elems = ctx.entries.asScala.map(e => this.visitChecked(e)).toVector;
      Expr(ExprKind.MakeVec(elems), Types.Vec(Types.unknown), ctx)
    }

    override def visitMakeStruct(ctx: MakeStructContext): Expr = {
      val elems = ctx.entries.asScala.map(e => this.visitChecked(e)).toVector;
      Expr(ExprKind.MakeStruct(elems), Types.Struct(elems.map(_ => Types.unknown).toVector), ctx)
    }

    override def visitIf(ctx: IfContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val cond = this.visitChecked(ctx.cond);
      val onTrue = this.visitChecked(ctx.onTrue);
      val onFalse = this.visitChecked(ctx.onFalse);
      Expr(ExprKind.If(cond, onTrue, onFalse), Types.unknown, ctx, annot)
    }

    override def visitIterate(ctx: IterateContext): Expr = {
      val init = this.visitChecked(ctx.initial);
      val upF = this.visitChecked(ctx.updateFunc);
      Expr(ExprKind.Iterate(init, upF), Types.unknown, ctx)
    }

    override def visitSelect(ctx: SelectContext): Expr = {
      val cond = this.visitChecked(ctx.cond);
      val onTrue = this.visitChecked(ctx.onTrue);
      val onFalse = this.visitChecked(ctx.onFalse);
      Expr(ExprKind.Select(cond, onTrue, onFalse), Types.unknown, ctx)
    }

    override def visitBroadcast(ctx: ArcParser.BroadcastContext): Expr = {
      val expr = this.visitChecked(ctx.valueExpr());
      Expr(ExprKind.Broadcast(expr), Types.Simd(Types.unknown), ctx)
    }

    override def visitSerialize(ctx: ArcParser.SerializeContext): Expr = {
      val expr = this.visitChecked(ctx.valueExpr());
      Expr(ExprKind.Serialize(expr), Types.Vec(Types.I8), ctx)
    }

    override def visitDeserialize(ctx: ArcParser.DeserializeContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val expr = this.visitChecked(ctx.valueExpr());
      val ty = TypeVisitor.visitChecked(ctx.`type`());
      Expr(ExprKind.Deserialize(ty, expr), ty, ctx, annot)
    }

    override def visitCUDF(ctx: CUDFContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val udf = this.visitChecked(ctx.cudfExpr());
      if (annot.isDefined) {
        Expr(udf.kind, udf.ty, ctx, annot)
      } else {
        udf
      }
    }

    override def visitPointerUDF(ctx: ArcParser.PointerUDFContext): Expr = {
      val pointer = this.visitChecked(ctx.funcPointer);
      val returnType = TypeVisitor.visitChecked(ctx.returnType);
      val args = ctx.functionParams().params.asScala.map { e =>
        this.visitChecked(e)
      }.toVector;
      Expr(ExprKind.CUDF(Right(pointer), args, returnType), returnType, ctx)
    }

    override def visitNameUDF(ctx: ArcParser.NameUDFContext): Expr = {
      val name = tokenToSymbol(ctx.name);
      val returnType = TypeVisitor.visitChecked(ctx.returnType);
      val args = ctx.functionParams().params.asScala.map { e =>
        this.visitChecked(e)
      }.toVector;
      Expr(ExprKind.CUDF(Left(name), args, returnType), returnType, ctx)
    }

    override def visitMerge(ctx: MergeContext): Expr = {
      val builder = this.visitChecked(ctx.builder);
      val value = this.visitChecked(ctx.value);
      Expr(ExprKind.Merge(builder, value), builder.ty, ctx)
    }

    override def visitResult(ctx: ResultContext): Expr = {
      val builder = this.visitChecked(ctx.valueExpr());
      Expr(ExprKind.Result(builder), Types.unknown, ctx)
    }
    override def visitNewAppender(ctx: NewAppenderContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val argO = Option(ctx.arg);
      val arg = argO.map(this.visitChecked(_));
      val elemTy = TypeVisitor.visitChecked(ctx.elemT, allowNull = true);
      val ty = Types.Builders.Appender(elemTy, annot);
      Expr(ExprKind.NewBuilder(ty, arg), ty, ctx)
    }

    override def visitNewStreamAppender(ctx: NewStreamAppenderContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val arg = None;
      val elemTy = TypeVisitor.visitChecked(ctx.elemT, allowNull = true);
      val ty = Types.Builders.StreamAppender(elemTy, annot);
      Expr(ExprKind.NewBuilder(ty, arg), ty, ctx)
    }

    override def visitNewMerger(ctx: NewMergerContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val argO = Option(ctx.arg);
      val arg = argO.map(this.visitChecked(_));
      val elemTy = TypeVisitor.visitChecked(ctx.elemT);
      val opTy = OpVisitor.visitChecked(ctx.commutativeBinop());
      val ty = Types.Builders.Merger(elemTy, opTy, annot);
      Expr(ExprKind.NewBuilder(ty, arg), ty, ctx)
    }

    override def visitNewDictMerger(ctx: NewDictMergerContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val argO = Option(ctx.arg);
      val arg = argO.map(this.visitChecked(_));
      val keyTy = TypeVisitor.visitChecked(ctx.keyT);
      val valueTy = TypeVisitor.visitChecked(ctx.valueT);
      val opTy = OpVisitor.visitChecked(ctx.commutativeBinop());
      val ty = Types.Builders.DictMerger(keyTy, valueTy, opTy, annot);
      Expr(ExprKind.NewBuilder(ty, arg), ty, ctx)
    }
    override def visitNewVecMerger(ctx: NewVecMergerContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val argO = Option(ctx.arg);
      val arg = argO.map(this.visitChecked(_));
      val elemTy = TypeVisitor.visitChecked(ctx.elemT);
      val opTy = OpVisitor.visitChecked(ctx.commutativeBinop());
      val ty = Types.Builders.VecMerger(elemTy, opTy, annot);
      Expr(ExprKind.NewBuilder(ty, arg), ty, ctx)
    }
    override def visitNewGroupMerger(ctx: NewGroupMergerContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val argO = Option(ctx.arg);
      val arg = argO.map(this.visitChecked(_));
      val keyTy = TypeVisitor.visitChecked(ctx.keyT, allowNull = true);
      val valueTy = TypeVisitor.visitChecked(ctx.valueT, allowNull = true);
      val ty = Types.Builders.GroupMerger(keyTy, valueTy, annot);
      Expr(ExprKind.NewBuilder(ty, arg), ty, ctx)
    }

    override def visitBinaryFunction(ctx: BinaryFunctionContext): Expr = {
      import ArcLexer._;

      val kind = ctx.fun.getType match {
        case TMin => BinOpKind.Min
        case TMax => BinOpKind.Max
        case TPow => BinOpKind.Pow
      };
      val left = this.visitChecked(ctx.left);
      val right = this.visitChecked(ctx.right);
      Expr(ExprKind.BinOp(kind, left, right), Types.unknown, ctx)
    }

    override def visitApplication(ctx: ArcParser.ApplicationContext): Expr = {
      val func = this.visitChecked(ctx.operatorExpr());
      val args = ctx.functionParams().params.asScala.map { e =>
        this.visitChecked(e)
      }.toVector;
      Expr(ExprKind.Application(func, args), Types.unknown, ctx)
    }

    override def visitProjection(ctx: ArcParser.ProjectionContext): Expr = {
      val struct = this.visitChecked(ctx.operatorExpr());
      val index = BigInt(ctx.TIndex().getText.substring(1), 10).toInt;
      Expr(ExprKind.Projection(struct, index), Types.unknown, ctx)
    }

    override def visitAscription(ctx: ArcParser.AscriptionContext): Expr = {
      val expr = this.visitChecked(ctx.operatorExpr());
      val ty = TypeVisitor.visitChecked(ctx.`type`());
      Expr(ExprKind.Ascription(expr, ty), ty, ctx)
    }

    override def visitProduct(ctx: ArcParser.ProductContext): Expr = {
      import ArcLexer._;

      val kind = ctx.op.getType match {
        case TStar    => BinOpKind.Mult
        case TSlash   => BinOpKind.Div
        case TPercent => BinOpKind.Modulo
      };
      val left = this.visitChecked(ctx.left);
      val right = this.visitChecked(ctx.right);
      Expr(ExprKind.BinOp(kind, left, right), Types.unknown, ctx)
    }

    override def visitSum(ctx: ArcParser.SumContext): Expr = {
      import ArcLexer._;

      val kind = ctx.op.getType match {
        case TPlus  => BinOpKind.Add
        case TMinus => BinOpKind.Sub
      };
      val left = this.visitChecked(ctx.left);
      val right = this.visitChecked(ctx.right);
      Expr(ExprKind.BinOp(kind, left, right), Types.unknown, ctx)
    }

    override def visitComparison(ctx: ArcParser.ComparisonContext): Expr = {
      import ArcLexer._;

      val kind = ctx.op.getType match {
        case TLessThan    => BinOpKind.LessThan
        case TGreaterThan => BinOpKind.GreaterThan
        case TLEq         => BinOpKind.LEq
        case TGEq         => BinOpKind.GEq
      };
      val left = this.visitChecked(ctx.left);
      val right = this.visitChecked(ctx.right);
      Expr(ExprKind.BinOp(kind, left, right), Types.unknown, ctx)
    }

    override def visitEquality(ctx: ArcParser.EqualityContext): Expr = {
      import ArcLexer._;

      val kind = ctx.op.getType match {
        case TEqualEqual => BinOpKind.Equals
        case TNotEqual   => BinOpKind.NEq
      };
      val left = this.visitChecked(ctx.left);
      val right = this.visitChecked(ctx.right);
      Expr(ExprKind.BinOp(kind, left, right), Types.unknown, ctx)
    }

    override def visitBitwiseXor(ctx: ArcParser.BitwiseXorContext): Expr = {
      val kind = BinOpKind.Xor;
      val left = this.visitChecked(ctx.left);
      val right = this.visitChecked(ctx.right);
      Expr(ExprKind.BinOp(kind, left, right), Types.unknown, ctx)
    }

    override def visitBitwiseOr(ctx: ArcParser.BitwiseOrContext): Expr = {
      val kind = BinOpKind.Or;
      val left = this.visitChecked(ctx.left);
      val right = this.visitChecked(ctx.right);
      Expr(ExprKind.BinOp(kind, left, right), Types.unknown, ctx)
    }

    override def visitBitwiseAnd(ctx: ArcParser.BitwiseAndContext): Expr = {
      val kind = BinOpKind.And;
      val left = this.visitChecked(ctx.left);
      val right = this.visitChecked(ctx.right);
      Expr(ExprKind.BinOp(kind, left, right), Types.unknown, ctx)
    }

    override def visitLogicalOr(ctx: ArcParser.LogicalOrContext): Expr = {
      val kind = BinOpKind.LogicalOr;
      val left = this.visitChecked(ctx.left);
      val right = this.visitChecked(ctx.right);
      Expr(ExprKind.BinOp(kind, left, right), Types.unknown, ctx)
    }

    override def visitLogicalAnd(ctx: ArcParser.LogicalAndContext): Expr = {
      val kind = BinOpKind.LogicalAnd;
      val left = this.visitChecked(ctx.left);
      val right = this.visitChecked(ctx.right);
      Expr(ExprKind.BinOp(kind, left, right), Types.unknown, ctx)
    }

    override def visitI8Lit(ctx: I8LitContext): Expr = {
      val raw = ctx.TI8Lit().getText;
      val num = extractInteger(raw);
      Expr(ExprKind.Literal.I8(raw, num.toInt), Types.I8, ctx)
    }
    override def visitI16Lit(ctx: I16LitContext): Expr = {
      val raw = ctx.TI16Lit().getText;
      val num = extractInteger(raw);
      Expr(ExprKind.Literal.I16(raw, num.toInt), Types.I16, ctx)
    }
    override def visitI32Lit(ctx: I32LitContext): Expr = {
      val raw = ctx.TI32Lit().getText;
      val num = extractInteger(raw);
      Expr(ExprKind.Literal.I32(raw, num.toInt), Types.I32, ctx)
    }
    override def visitI64Lit(ctx: I64LitContext): Expr = {
      val raw = ctx.TI64Lit().getText;
      val num = extractInteger(raw);
      Expr(ExprKind.Literal.I64(raw, num.toLong), Types.I64, ctx)
    }

    override def visitF32Lit(ctx: F32LitContext): Expr = {
      val raw = ctx.TF32Lit().getText;
      val num = raw.toFloat;
      Expr(ExprKind.Literal.F32(raw, num), Types.F32, ctx)
    }
    override def visitF64Lit(ctx: F64LitContext): Expr = {
      val raw = ctx.TF64Lit().getText;
      val num = raw.toDouble;
      Expr(ExprKind.Literal.F64(raw, num), Types.F64, ctx)
    }

    override def visitBoolLit(ctx: BoolLitContext): Expr = {
      val raw = ctx.TBoolLit().getText;
      val b = raw match {
        case "true"  => true
        case "false" => false
      };
      Expr(ExprKind.Literal.Bool(raw, b), Types.Bool, ctx)
    }

    override def visitStringLit(ctx: StringLitContext): Expr = {
      val raw = ctx.TStringLit().getText;
      val s = raw.substring(1, raw.length() - 1); // TODO replace escapes
      Expr(ExprKind.Literal.StringL(raw, s), Types.StringT, ctx)
    }

    val binExpr = raw"0b([01]+)(?:[cClL]|si)?".r;
    val hexExpr = raw"0x([0-9a-fA-F]+)(?:[cClL]|si)?".r;
    val decExpr = raw"([0-9]+)(?:[cClL]|si)?".r;

    private def extractInteger(s: String): BigInt = {
      s match {
        case binExpr(core) => BigInt(core, 2)
        case hexExpr(core) => BigInt(core, 16)
        case decExpr(core) => BigInt(core, 10)
      }
    }
  }

  object AnnotationVisitor extends ArcBaseVisitor[Annotations] {
    def visitChecked(tree: ParseTree): Option[Annotations] = {
      if (tree == null) {
        None
      } else {
        val e = this.visit(tree);
        assert(e != null, s"Visiting a sub-tree returned null:\n${tree.toStringTree(parser)}");
        Some(e)
      }
    }

    override def visitAnnotations(ctx: AnnotationsContext): Annotations = {
      val entries = ctx.entries.asScala.flatMap(this.visitChecked(_)).foldLeft(Vector.empty[(String, Any)]) {
        case (acc, a) => acc :+ a.params(0)
      };
      Annotations(entries)
    }

    override def visitIdPair(ctx: IdPairContext): Annotations = {
      val key = ctx.name.getText;
      val value = ctx.value.getText;
      Annotations(Vector(key -> value))
    }

    override def visitLiteralPair(ctx: LiteralPairContext): Annotations = {
      val key = ctx.name.getText;
      val value = ExprVisitor.visitChecked(ctx.value).kind;
      Annotations(Vector(key -> value))
    }
  }

  object OpVisitor extends ArcBaseVisitor[OpType] {
    def visitChecked(tree: ParseTree): OpType = {
      val e = this.visit(tree);
      assert(e != null, s"Visiting a sub-tree returned null:\n${tree.toStringTree()}");
      e
    }

    override def visitSumOp(ctx: SumOpContext): OpType = OpTypes.Sum;
    override def visitProductOp(ctx: ProductOpContext): OpType = OpTypes.Product;
    override def visitMaxOp(ctx: MaxOpContext): OpType = OpTypes.Max;
    override def visitMinOp(ctx: MinOpContext): OpType = OpTypes.Min;

  }

  object IterVisitor extends ArcBaseVisitor[Iter] {
    def visitChecked(tree: ParseTree): Iter = {
      val e = this.visit(tree);
      assert(e != null, s"Visiting a sub-tree returned null:\n${tree.toStringTree(parser)}");
      e
    }

    def tokenToIterKind(t: Token): IterKind.IterKind = {
      import ArcLexer._;
      import IterKind._;

      t.getType match {
        case TScalarIter => ScalarIter
        case TSimdIter   => SimdIter
        case TFringeIter => FringeIter
        case TNdIter     => NdIter
        case TRangeIter  => RangeIter
        case _           => UnknownIter
      }
    }

    override def visitSimpleIter(ctx: ArcParser.SimpleIterContext): Iter = {
      val kind = tokenToIterKind(ctx.iter);
      val data = ExprVisitor.visitChecked(ctx.data);
      Iter(kind, data)
    }

    override def visitFourIter(ctx: ArcParser.FourIterContext): Iter = {
      val kind = tokenToIterKind(ctx.iter);
      val data = ExprVisitor.visitChecked(ctx.data);
      val start = ExprVisitor.visitChecked(ctx.start);
      val end = ExprVisitor.visitChecked(ctx.end);
      val stride = ExprVisitor.visitChecked(ctx.stride);
      Iter(kind = kind, data = data, start = Some(start), end = Some(end), stride = Some(stride))
    }

    override def visitSixIter(ctx: ArcParser.SixIterContext): Iter = {
      val kind = IterKind.NdIter;
      val data = ExprVisitor.visitChecked(ctx.data);
      val start = ExprVisitor.visitChecked(ctx.start);
      val end = ExprVisitor.visitChecked(ctx.end);
      val stride = ExprVisitor.visitChecked(ctx.stride);
      val shape = ExprVisitor.visitChecked(ctx.shape);
      val strides = ExprVisitor.visitChecked(ctx.strides);
      Iter(kind = kind, data = data,
        start = Some(start), end = Some(end),
        stride = Some(stride), shape = Some(shape), strides = Some(strides))
    }

    override def visitRangeIter(ctx: ArcParser.RangeIterContext): Iter = {
      val kind = IterKind.RangeIter;
      val dummyData = Expr(ExprKind.MakeVec(Vector.empty), Types.Vec(Types.I64), ctx);
      val start = ExprVisitor.visitChecked(ctx.start);
      val end = ExprVisitor.visitChecked(ctx.end);
      val stride = ExprVisitor.visitChecked(ctx.stride);
      Iter(kind = kind, data = dummyData, start = Some(start), end = Some(end), stride = Some(stride))
    }

    override def visitUnkownIter(ctx: ArcParser.UnkownIterContext): Iter = {
      val data = ExprVisitor.visitChecked(ctx.valueExpr());
      Iter(IterKind.UnknownIter, data)
    }
  }

  object TypeVisitor extends ArcBaseVisitor[Type] {
    def visitChecked(tree: ParseTree, allowNull: Boolean = false): Type = {
      if (allowNull && tree == null) {
        Types.unknown // just return a type variable
      } else {
        assert(tree != null, s"Can't extract type from null-tree");
        val e = this.visit(tree);
        assert(e != null, s"Visiting a sub-tree returned null:\n${tree.toStringTree(parser)}");
        e
      }
    }

    def tokenToScalar(t: Token): Option[Types.Scalar] = {
      import ArcLexer._;
      Try(t.getType match {
        case TI8   => Types.I8
        case TI16  => Types.I16
        case TI32  => Types.I32
        case TI64  => Types.I64
        case TU8   => Types.U8
        case TU16  => Types.U16
        case TU32  => Types.U32
        case TU64  => Types.U64
        case TF32  => Types.F32
        case TF64  => Types.F64
        case TBool => Types.Bool
        case TUnit => Types.UnitT
      }).toOption
    }

    override def visitI8(ctx: I8Context): Type = Types.I8;
    override def visitI16(ctx: I16Context): Type = Types.I16;
    override def visitI32(ctx: I32Context): Type = Types.I32;
    override def visitI64(ctx: I64Context): Type = Types.I64;
    override def visitU8(ctx: U8Context): Type = Types.U8;
    override def visitU16(ctx: U16Context): Type = Types.U16;
    override def visitU32(ctx: U32Context): Type = Types.U32;
    override def visitU64(ctx: U64Context): Type = Types.U64;
    override def visitF32(ctx: F32Context): Type = Types.F32;
    override def visitF64(ctx: F64Context): Type = Types.F64;
    override def visitBool(ctx: BoolContext): Type = Types.Bool;
    override def visitUnitT(ctx: UnitTContext): Type = Types.UnitT;
    override def visitStringT(ctx: StringTContext): Type = Types.StringT;

    override def visitSimd(ctx: SimdContext): Type = {
      val elemT = this.visitChecked(ctx.elemT);
      Types.Simd(elemT)
    }

    override def visitVec(ctx: VecContext): Type = {
      val elemT = this.visitChecked(ctx.elemT);
      Types.Vec(elemT)
    }

    override def visitStream(ctx: StreamContext): Type = {
      val elemT = this.visitChecked(ctx.elemT);
      Types.Stream(elemT)
    }

    override def visitDict(ctx: DictContext): Type = {
      val keyT = this.visitChecked(ctx.keyT);
      val valueT = this.visitChecked(ctx.valueT);
      Types.Dict(keyT, valueT)
    }

    override def visitStruct(ctx: StructContext): Type = {
      val elemTypes = ctx.types.asScala.map(t => TypeVisitor.visitChecked(t)).toVector;
      Types.Struct(elemTypes)
    }

    override def visitUnitFunction(ctx: ArcParser.UnitFunctionContext): Type = {
      val returnTy = this.visitChecked(ctx.returnT);
      Types.Function(Vector.empty, returnTy)
    }

    override def visitParamFunction(ctx: ArcParser.ParamFunctionContext): Type = {
      val params = ctx.paramTypes.asScala.map(this.visitChecked(_, false)).toVector;
      val returnTy = this.visitChecked(ctx.returnT);
      Types.Function(params, returnTy)
    }

    override def visitAppender(ctx: ArcParser.AppenderContext): Type = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val elemTy = TypeVisitor.visitChecked(ctx.elemT);
      Types.Builders.Appender(elemTy, annot)
    }

    override def visitStreamAppender(ctx: ArcParser.StreamAppenderContext): Type = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val elemTy = TypeVisitor.visitChecked(ctx.elemT);
      Types.Builders.StreamAppender(elemTy, annot)
    }

    override def visitMerger(ctx: ArcParser.MergerContext): Type = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val elemTy = TypeVisitor.visitChecked(ctx.elemT);
      val opTy = OpVisitor.visitChecked(ctx.commutativeBinop());
      Types.Builders.Merger(elemTy, opTy, annot)
    }

    override def visitDictMerger(ctx: ArcParser.DictMergerContext): Type = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val keyTy = TypeVisitor.visitChecked(ctx.keyT);
      val valueTy = TypeVisitor.visitChecked(ctx.valueT);
      val opTy = OpVisitor.visitChecked(ctx.commutativeBinop());
      Types.Builders.DictMerger(keyTy, valueTy, opTy, annot)
    }

    override def visitGroupMerger(ctx: ArcParser.GroupMergerContext): Type = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val keyTy = TypeVisitor.visitChecked(ctx.keyT);
      val valueTy = TypeVisitor.visitChecked(ctx.valueT);
      Types.Builders.GroupMerger(keyTy, valueTy, annot)
    }

    override def visitVecMerger(ctx: ArcParser.VecMergerContext): Type = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations());
      val elemTy = TypeVisitor.visitChecked(ctx.elemT);
      val opTy = OpVisitor.visitChecked(ctx.commutativeBinop());
      Types.Builders.VecMerger(elemTy, opTy, annot)
    }

    override def visitTypeVariable(ctx: ArcParser.TypeVariableContext): Type = {
      Types.unknown // ignore the annotated number to avoid clashes
    }
  }
}

object ASTTranslator {
  import AST._;
  import ArcParser._;

  def apply(parser: ArcParser): ASTTranslator = new ASTTranslator(parser);

}
