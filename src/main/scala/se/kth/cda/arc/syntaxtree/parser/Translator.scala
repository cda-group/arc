package se.kth.cda.arc.syntaxtree.parser

import org.antlr.v4.runtime.Token
import org.antlr.v4.runtime.tree.ParseTree
import se.kth.cda.arc._
import se.kth.cda.arc.syntaxtree.{MergeOp, Type}

import scala.collection.JavaConverters._
import scala.util.Try
import scala.util.matching.Regex

final case class Translator(parser: ArcParser) {

  import ArcParser._
  import se.kth.cda.arc.syntaxtree.AST._

  def program(): Program = ProgramVisitor.visitChecked(parser.program())

  def expr(): Expr = ExprVisitor.visitChecked(parser.expr())

  def macros(): List[Macro] = MacrosVisitor.visitChecked(parser.macros())

  def `type`(): Type = TypeVisitor.visitChecked(parser.`type`())

  def translate(tree: ArcParser.ProductContext): Program = ProgramVisitor.visitChecked(tree)

  def translate(tree: ArcParser.ExprContext): Expr = ExprVisitor.visitChecked(tree)

  def translate(tree: ArcParser.MacroContext): List[Macro] = MacrosVisitor.visitChecked(tree)

  def translate(tree: ArcParser.TypeContext): Type = TypeVisitor.visitChecked(tree)

  private def tokenToSymbol(t: Token): Symbol = Symbol(t.getText, Some(t))

  private def annotToType(ctx: TypeAnnotContext): Option[Type] = Option(ctx).map(TypeVisitor.visitChecked(_))

  private def annotToTypeBound(ctx: TypeAnnotContext): Type = annotToType(ctx).getOrElse(Type.unknown)

  object ProgramVisitor extends ArcBaseVisitor[Program] {

    def visitChecked(tree: ParseTree): Program =
      Option(this.visit(tree))
        .getOrElse {
          throw new AssertionError(s"Visiting a sub-tree returned null:\n${tree.toStringTree(parser)}")
        }

    override def visitProgram(ctx: ProgramContext): Program =
      Program(
        macros = MacrosVisitor.visitMacros(ctx.macros()),
        expr = ExprVisitor.visitExpr(ctx.expr()),
        ctx
      )
  }

  object MacrosVisitor extends ArcBaseVisitor[List[Macro]] {

    def visitChecked(tree: ParseTree): List[Macro] =
      Option(this.visit(tree))
        .getOrElse {
          throw new AssertionError(s"Visiting a sub-tree returned null:\n${tree.toStringTree(parser)}")
        }

    override def visitMacros(ctx: MacrosContext): List[Macro] = ctx.`macro`().asScala.flatMap(visitChecked(_)).toList

    override def visitMacro(ctx: MacroContext): List[Macro] =
      List(
        Macro(
          symbol = tokenToSymbol(ctx.name),
          params = ctx.macroParams().names.asScala.map(tokenToSymbol).toVector,
          body = ExprVisitor.visitChecked(ctx.body),
          ctx
        )
      )
  }

  object ExprVisitor extends ArcBaseVisitor[Expr] {

    def visitChecked(tree: ParseTree): Expr =
      Option(this.visit(tree))
        .getOrElse {
          throw new AssertionError(s"Visiting a sub-tree returned null:\n${tree.toStringTree(parser)}")
        }

    override def visitParenExpr(ctx: ParenExprContext): Expr = this.visitChecked(ctx.expr())

    override def visitLetExpr(ctx: LetExprContext): Expr = {
      val body = this.visitChecked(ctx.body)
      Expr(
        kind = ExprKind.Let(
          symbol = tokenToSymbol(ctx.name),
          bindingTy = annotToType(ctx.typeAnnot()).getOrElse(Type.unknown),
          value = this.visitChecked(ctx.value),
          body = body
        ),
        ty = body.ty,
        ctx
      )
    }

    override def visitUnitLambda(ctx: UnitLambdaContext): Expr =
      Expr(
        kind = ExprKind.Lambda(
          Vector.empty,
          body = this.visitChecked(ctx.body)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitParamLambda(ctx: ParamLambdaContext): Expr =
      Expr(
        kind = ExprKind.Lambda(
          params = ctx
            .lambdaParams()
            .param()
            .asScala
            .map { p =>
              Parameter(
                symbol = tokenToSymbol(p.name),
                ty = annotToTypeBound(p.typeAnnot())
              )
            }
            .toVector,
          body = this.visitChecked(ctx.body)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitIdent(ctx: IdentContext): Expr =
      Expr(
        kind = ExprKind.Ident(
          symbol = tokenToSymbol(ctx.TIdentifier().getSymbol)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitCast(ctx: ArcParser.CastContext): Expr = {
      val Some(ty) = TypeVisitor.tokenToScalar(ctx.scalar)
      Expr(
        kind = ExprKind.Cast(
          ty,
          expr = this.visitChecked(ctx.valueExpr())
        ),
        ty,
        ctx
      )
    }

    override def visitToVec(ctx: ArcParser.ToVecContext): Expr =
      Expr(
        kind = ExprKind.ToVec(
          expr = this.visitChecked(ctx.valueExpr())
        ),
        ty = Type.Vec(
          elemTy = Type.unknown
        ),
        ctx
      )

    override def visitZip(ctx: ArcParser.ZipContext): Expr =
      Expr(
        kind = ExprKind.Zip(
          params = ctx.functionParams().params.asScala.map(this.visitChecked).toVector
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitHash(ctx: ArcParser.HashContext): Expr =
      Expr(
        kind = ExprKind.Hash(
          params = ctx.functionParams().params.asScala.map(this.visitChecked).toVector
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitFor(ctx: ArcParser.ForContext): Expr = {
      val builder = this.visitChecked(ctx.builder)
      Expr(
        kind = ExprKind.For(
          iterator = IterVisitor.visitChecked(ctx.iterator()),
          builder,
          body = this.visitChecked(ctx.body)
        ),
        ty = builder.ty,
        ctx,
        annotations = AnnotationVisitor.visitChecked(ctx.annotations())
      )
    }

    override def visitLen(ctx: ArcParser.LenContext): Expr =
      Expr(
        kind = ExprKind.Len(
          expr = this.visitChecked(ctx.valueExpr())
        ),
        ty = Type.I64,
        ctx
      )

    override def visitLookup(ctx: ArcParser.LookupContext): Expr =
      Expr(
        kind = ExprKind.Lookup(
          data = this.visitChecked(ctx.data),
          key = this.visitChecked(ctx.key)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitSlice(ctx: ArcParser.SliceContext): Expr = {
      val data = this.visitChecked(ctx.data)
      Expr(
        kind = ExprKind.Slice(
          data,
          index = this.visitChecked(ctx.index),
          size = this.visitChecked(ctx.size)
        ),
        ty = data.ty,
        ctx
      )
    }

    override def visitSort(ctx: ArcParser.SortContext): Expr = {
      val data = this.visitChecked(ctx.data)
      Expr(
        kind = ExprKind.Sort(
          data,
          keyFunc = this.visitChecked(ctx.keyFunc)
        ),
        ty = data.ty,
        ctx
      )
    }

    override def visitDrain(ctx: ArcParser.DrainContext): Expr = {
      val sink = this.visitChecked(ctx.sink)
      Expr(
        kind = ExprKind.Drain(
          source = this.visitChecked(ctx.source),
          sink,
        ),
        ty = sink.ty,
        ctx
      )
    }

    override def visitNegate(ctx: NegateContext): Expr = {
      val expr = this.visitChecked(ctx.operatorExpr())
      Expr(
        kind = ExprKind.Negate(
          expr
        ),
        ty = expr.ty,
        ctx
      )
    }

    override def visitNot(ctx: NotContext): Expr =
      Expr(
        kind = ExprKind.Not(
          expr = this.visitChecked(ctx.operatorExpr())
        ),
        ty = Type.Bool,
        ctx
      )

    override def visitUnaryOp(ctx: UnaryOpContext): Expr = {
      import ArcLexer._
      import UnaryOpKind._
      val expr = this.visitChecked(ctx.valueExpr())
      Expr(
        ExprKind.UnaryOp(
          kind = ctx.op.getType match {
            case TExp => Exp
            case TSin => Sin
            case TCos => Cos
            case TTan => Tan
            case TASin => ASin
            case TACos => ACos
            case TATan => ATan
            case TSinh => Sinh
            case TCosh => Cosh
            case TTanh => Tanh
            case TLog => Log
            case TErf => Erf
            case TSqrt => Sqrt
          },
          expr
        ),
        ty = expr.ty,
        ctx
      )
    }

    override def visitMakeVec(ctx: MakeVecContext): Expr =
      Expr(
        kind = ExprKind.MakeVec(
          elems = ctx.entries.asScala.map(this.visitChecked).toVector
        ),
        ty = Type.Vec(
          elemTy = Type.unknown
        ),
        ctx
      )

    override def visitMakeStruct(ctx: MakeStructContext): Expr = {
      val elems = ctx.entries.asScala.map(this.visitChecked).toVector
      Expr(
        kind = ExprKind.MakeStruct(elems),
        ty = Type.Struct(
          elemTys = elems.map(_ => Type.unknown)
        ),
        ctx
      )
    }

    override def visitIf(ctx: IfContext): Expr =
      Expr(
        kind = ExprKind.If(
          cond = this.visitChecked(ctx.cond),
          onTrue = this.visitChecked(ctx.onTrue),
          onFalse = this.visitChecked(ctx.onFalse)
        ),
        ty = Type.unknown,
        ctx,
        annotations = AnnotationVisitor.visitChecked(ctx.annotations())
      )

    override def visitIterate(ctx: IterateContext): Expr =
      Expr(
        kind = ExprKind.Iterate(
          initial = this.visitChecked(ctx.initial),
          updateFunc = this.visitChecked(ctx.updateFunc)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitSelect(ctx: SelectContext): Expr =
      Expr(
        kind = ExprKind.Select(
          cond = this.visitChecked(ctx.cond),
          onTrue = this.visitChecked(ctx.onTrue),
          onFalse = this.visitChecked(ctx.onFalse)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitBroadcast(ctx: ArcParser.BroadcastContext): Expr = {
      Expr(
        kind = ExprKind.Broadcast(
          expr = this.visitChecked(ctx.valueExpr())
        ),
        ty = Type.Simd(
          elemTy = Type.unknown
        ),
        ctx
      )
    }

    override def visitSerialize(ctx: ArcParser.SerializeContext): Expr =
      Expr(
        kind = ExprKind.Serialize(
          expr = this.visitChecked(ctx.valueExpr())
        ),
        ty = Type.Vec(
          elemTy = Type.I8
        ),
        ctx
      )

    override def visitDeserialize(ctx: ArcParser.DeserializeContext): Expr = {
      val ty = TypeVisitor.visitChecked(ctx.`type`())
      Expr(
        kind = ExprKind.Deserialize(
          ty,
          expr = this.visitChecked(ctx.valueExpr())
        ),
        ty,
        ctx,
        annotations = AnnotationVisitor.visitChecked(ctx.annotations())
      )
    }

    override def visitCUDF(ctx: CUDFContext): Expr = {
      val annotations = AnnotationVisitor.visitChecked(ctx.annotations())
      val udf = this.visitChecked(ctx.cudfExpr())
      if (annotations.isDefined) {
        Expr(
          kind = udf.kind,
          ty = udf.ty,
          ctx,
          annotations
        )
      } else {
        udf
      }
    }

    override def visitPointerUDF(ctx: ArcParser.PointerUDFContext): Expr = {
      val returnType = TypeVisitor.visitChecked(ctx.returnType)
      Expr(
        kind = ExprKind.CUDF(
          reference = Right(this.visitChecked(ctx.funcPointer)),
          args = ctx
            .functionParams()
            .params
            .asScala
            .map(this.visitChecked)
            .toVector,
          returnTy = TypeVisitor.visitChecked(ctx.returnType)
        ),
        ty = returnType,
        ctx
      )
    }

    override def visitNameUDF(ctx: ArcParser.NameUDFContext): Expr = {
      val returnType = TypeVisitor.visitChecked(ctx.returnType)
      Expr(
        kind = ExprKind.CUDF(
          reference = Left(tokenToSymbol(ctx.name)),
          args = ctx
            .functionParams()
            .params
            .asScala
            .map(this.visitChecked)
            .toVector,
          returnType
        ),
        ty = returnType,
        ctx
      )
    }

    override def visitMerge(ctx: MergeContext): Expr = {
      val builder = this.visitChecked(ctx.builder)
      Expr(
        kind = ExprKind.Merge(
          builder,
          value = this.visitChecked(ctx.value)
        ),
        ty = builder.ty,
        ctx
      )
    }

    override def visitResult(ctx: ResultContext): Expr = {
      Expr(
        kind = ExprKind.Result(
          expr = this.visitChecked(ctx.valueExpr())
        ),
        ty = Type.unknown,
        ctx
      )
    }

    override def visitNewAppender(ctx: NewAppenderContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations())
      val elemTy = TypeVisitor.visitChecked(ctx.elemT, allowNull = true)
      val ty = Type.Builder.Appender(elemTy, annot)
      Expr(
        kind = ExprKind.NewBuilder(
          ty,
          args = Option(ctx.arg).map(this.visitChecked).toVector
        ),
        ty,
        ctx
      )
    }

    override def visitNewStreamAppender(ctx: NewStreamAppenderContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations())
      val elemTy = TypeVisitor.visitChecked(ctx.elemT, allowNull = true)
      val ty = Type.Builder.StreamAppender(elemTy, annot)
      Expr(
        kind = ExprKind.NewBuilder(
          ty,
          args = Vector.empty
        ),
        ty,
        ctx
      )
    }

    override def visitNewMerger(ctx: NewMergerContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations())
      val elemTy = TypeVisitor.visitChecked(ctx.elemT)
      val opTy = OpVisitor.visitChecked(ctx.commutativeBinop())
      val ty = Type.Builder.Merger(elemTy, opTy, annot)
      Expr(
        kind = ExprKind.NewBuilder(
          ty,
          args = Option(ctx.arg).map(this.visitChecked).toVector
        ),
        ty,
        ctx
      )
    }

    override def visitNewDictMerger(ctx: NewDictMergerContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations())
      val keyTy = TypeVisitor.visitChecked(ctx.keyT)
      val valueTy = TypeVisitor.visitChecked(ctx.valueT)
      val opTy = OpVisitor.visitChecked(ctx.commutativeBinop())
      val ty = Type.Builder.DictMerger(keyTy, valueTy, opTy, annot)
      Expr(
        kind = ExprKind.NewBuilder(
          ty,
          args = Option(ctx.arg).map(this.visitChecked).toVector
        ),
        ty,
        ctx
      )
    }

    override def visitNewVecMerger(ctx: NewVecMergerContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations())
      val elemTy = TypeVisitor.visitChecked(ctx.elemT)
      val opTy = OpVisitor.visitChecked(ctx.commutativeBinop())
      val ty = Type.Builder.VecMerger(elemTy, opTy, annot)
      Expr(
        kind = ExprKind.NewBuilder(
          ty,
          args = Option(ctx.arg).map(this.visitChecked).toVector
        ),
        ty,
        ctx
      )
    }

    override def visitNewGroupMerger(ctx: NewGroupMergerContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations())
      val keyTy = TypeVisitor.visitChecked(ctx.keyT, allowNull = true)
      val valueTy = TypeVisitor.visitChecked(ctx.valueT, allowNull = true)
      val ty = Type.Builder.GroupMerger(keyTy, valueTy, annot)
      Expr(
        kind = ExprKind.NewBuilder(
          ty,
          args = Option(ctx.arg).map(this.visitChecked).toVector
        ),
        ty,
        ctx
      )
    }

    override def visitNewWindower(ctx: NewWindowerContext): Expr = {
      val annot = AnnotationVisitor.visitChecked(ctx.annotations())
      val ty = Type.Builder.Windower(
        discTy = TypeVisitor.visitChecked(ctx.discT),
        aggrTy = TypeVisitor.visitChecked(ctx.aggrT),
        aggrMergeTy = Type.unknown,
        aggrResultTy = Type.unknown,
        annot)
      Expr(
        kind = ExprKind.NewBuilder(
          ty,
          args = Vector(
            ExprVisitor.visitChecked(ctx.assign),
            ExprVisitor.visitChecked(ctx.trigger),
            ExprVisitor.visitChecked(ctx.lower))
        ),
        ty,
        ctx
      )
    }

    override def visitBinaryFunction(ctx: BinaryFunctionContext): Expr =
      Expr(
        kind = ExprKind.BinOp(
          kind = ctx.fun.getType match {
            case ArcLexer.TMin => BinOpKind.Min
            case ArcLexer.TMax => BinOpKind.Max
            case ArcLexer.TPow => BinOpKind.Pow
          },
          lhs = this.visitChecked(ctx.left),
          rhs = this.visitChecked(ctx.right)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitApplication(ctx: ArcParser.ApplicationContext): Expr =
      Expr(
        kind = ExprKind.Application(
          expr = this.visitChecked(ctx.operatorExpr()),
          args = ctx
            .functionParams()
            .params
            .asScala
            .map(this.visitChecked)
            .toVector
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitProjection(ctx: ArcParser.ProjectionContext): Expr =
      Expr(
        kind = ExprKind.Projection(
          expr = this.visitChecked(ctx.operatorExpr()),
          index = BigInt(ctx.TIndex().getText.substring(1), decRadix).toInt
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitAscription(ctx: ArcParser.AscriptionContext): Expr = {
      val ty = TypeVisitor.visitChecked(ctx.`type`())
      Expr(
        kind = ExprKind.Ascription(
          expr = this.visitChecked(ctx.operatorExpr()),
          ty
        ),
        ty,
        ctx
      )
    }

    override def visitProduct(ctx: ArcParser.ProductContext): Expr =
      Expr(
        kind = ExprKind.BinOp(
          kind = ctx.op.getType match {
            case ArcLexer.TStar => BinOpKind.Mul
            case ArcLexer.TSlash => BinOpKind.Div
            case ArcLexer.TPercent => BinOpKind.Mod
          },
          lhs = this.visitChecked(ctx.left),
          rhs = this.visitChecked(ctx.right)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitSum(ctx: ArcParser.SumContext): Expr =
      Expr(
        kind = ExprKind.BinOp(
          kind = ctx.op.getType match {
            case ArcLexer.TPlus => BinOpKind.Add
            case ArcLexer.TMinus => BinOpKind.Sub
          },
          lhs = this.visitChecked(ctx.left),
          rhs = this.visitChecked(ctx.right)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitComparison(ctx: ArcParser.ComparisonContext): Expr =
      Expr(
        kind = ExprKind.BinOp(
          kind = ctx.op.getType match {
            case ArcLexer.TLessThan => BinOpKind.Lt
            case ArcLexer.TGreaterThan => BinOpKind.Gt
            case ArcLexer.TLEq => BinOpKind.LEq
            case ArcLexer.TGEq => BinOpKind.GEq
          },
          lhs = this.visitChecked(ctx.left),
          rhs = this.visitChecked(ctx.right)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitEquality(ctx: ArcParser.EqualityContext): Expr =
      Expr(
        kind = ExprKind.BinOp(
          kind = ctx.op.getType match {
            case ArcLexer.TEqualEqual => BinOpKind.Eq
            case ArcLexer.TNotEqual => BinOpKind.NEq
          },
          lhs = this.visitChecked(ctx.left),
          rhs = this.visitChecked(ctx.right)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitBitwiseXor(ctx: ArcParser.BitwiseXorContext): Expr =
      Expr(
        kind = ExprKind.BinOp(
          kind = BinOpKind.BwXor,
          lhs = this.visitChecked(ctx.left),
          rhs = this.visitChecked(ctx.right)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitBitwiseOr(ctx: ArcParser.BitwiseOrContext): Expr =
      Expr(
        kind = ExprKind.BinOp(
          kind = BinOpKind.BwOr,
          lhs = this.visitChecked(ctx.left),
          rhs = this.visitChecked(ctx.right)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitBitwiseAnd(ctx: ArcParser.BitwiseAndContext): Expr =
      Expr(
        kind = ExprKind.BinOp(
          kind = BinOpKind.BwAnd,
          lhs = this.visitChecked(ctx.left),
          rhs = this.visitChecked(ctx.right)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitLogicalOr(ctx: ArcParser.LogicalOrContext): Expr =
      Expr(
        kind = ExprKind.BinOp(
          kind = BinOpKind.Or,
          lhs = this.visitChecked(ctx.left),
          rhs = this.visitChecked(ctx.right)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitLogicalAnd(ctx: ArcParser.LogicalAndContext): Expr =
      Expr(
        kind = ExprKind.BinOp(
          kind = BinOpKind.And,
          lhs = this.visitChecked(ctx.left),
          rhs = this.visitChecked(ctx.right)
        ),
        ty = Type.unknown,
        ctx
      )

    override def visitI8Lit(ctx: I8LitContext): Expr = {
      val raw = ctx.TI8Lit().getText
      Expr(
        kind = ExprKind.Literal.I8(
          raw,
          value = extractInteger(raw).toInt
        ),
        ty = Type.I8,
        ctx
      )
    }

    override def visitI16Lit(ctx: I16LitContext): Expr = {
      val raw = ctx.TI16Lit().getText
      Expr(
        kind = ExprKind.Literal.I16(
          raw,
          value = extractInteger(raw).toInt
        ),
        ty = Type.I16,
        ctx
      )
    }

    override def visitI32Lit(ctx: I32LitContext): Expr = {
      val raw = ctx.TI32Lit().getText
      Expr(
        kind = ExprKind.Literal.I32(
          raw,
          value = extractInteger(raw).toInt
        ),
        ty = Type.I32,
        ctx
      )
    }

    override def visitI64Lit(ctx: I64LitContext): Expr = {
      val raw = ctx.TI64Lit().getText
      Expr(
        kind = ExprKind.Literal.I64(
          raw,
          value = extractInteger(raw).toLong
        ),
        ty = Type.I64,
        ctx
      )
    }

    override def visitF32Lit(ctx: F32LitContext): Expr = {
      val raw = ctx.TF32Lit().getText
      Expr(
        kind = ExprKind.Literal.F32(
          raw,
          value = raw.toFloat
        ),
        ty = Type.F32,
        ctx
      )
    }

    override def visitF64Lit(ctx: F64LitContext): Expr = {
      val raw = ctx.TF64Lit().getText
      Expr(
        kind = ExprKind.Literal.F64(
          raw,
          value = raw.toDouble
        ),
        ty = Type.F64,
        ctx
      )
    }

    override def visitBoolLit(ctx: BoolLitContext): Expr = {
      val raw = ctx.TBoolLit().getText
      Expr(
        kind = ExprKind.Literal.Bool(
          raw,
          value = raw match {
            case "true" => true
            case "false" => false
          }
        ),
        ty = Type.Bool,
        ctx
      )
    }

    override def visitStringLit(ctx: StringLitContext): Expr = {
      val raw = ctx.TStringLit().getText
      Expr(
        kind = ExprKind.Literal.StringL(
          raw,
          value = raw.substring(1, raw.length() - 1) // TODO replace escapes
        ),
        ty = Type.StringT,
        ctx
      )
    }

    val binExpr: Regex = raw"0b([01]+)(?:[cClL]|si)?".r
    val hexExpr: Regex = raw"0x([0-9a-fA-F]+)(?:[cClL]|si)?".r
    val decExpr: Regex = raw"([0-9]+)(?:[cClL]|si)?".r

    val binRadix: Int = 2
    val hexRadix: Int = 16
    val decRadix: Int = 10

    private def extractInteger(s: String): BigInt =
      s match {
        case binExpr(core) => BigInt(core, binRadix)
        case hexExpr(core) => BigInt(core, hexRadix)
        case decExpr(core) => BigInt(core, decRadix)
      }
  }

  object AnnotationVisitor extends ArcBaseVisitor[Annotations] {

    def visitChecked(tree: ParseTree): Option[Annotations] =
      if (tree == null) {
        None
      } else {
        Option(this.visit(tree))
          .orElse {
            throw new AssertionError(s"Visiting a sub-tree returned null:\n${tree.toStringTree(parser)}")
          }
      }

    override def visitAnnotations(ctx: AnnotationsContext): Annotations =
      Annotations(
        params = ctx.entries.asScala
          .flatMap(this.visitChecked)
          .foldLeft(Vector.empty[(String, Any)]) {
            case (acc, a) => acc :+ a.params(0)
          }
      )

    override def visitIdPair(ctx: IdPairContext): Annotations =
      Annotations(
        params = Vector(ctx.name.getText -> ctx.value.getText)
      )

    override def visitLiteralPair(ctx: LiteralPairContext): Annotations =
      Annotations(
        params = Vector(ctx.name.getText -> ExprVisitor.visitChecked(ctx.value).kind)
      )
  }

  object OpVisitor extends ArcBaseVisitor[MergeOp] {

    def visitChecked(tree: ParseTree): MergeOp =
      Option(this.visit(tree))
        .getOrElse(
          throw new AssertionError(s"Visiting a sub-tree returned null:\n${tree.toStringTree()}")
        )

    override def visitSumOp(ctx: SumOpContext): MergeOp = MergeOp.Sum

    override def visitProductOp(ctx: ProductOpContext): MergeOp = MergeOp.Product

    override def visitMaxOp(ctx: MaxOpContext): MergeOp = MergeOp.Max

    override def visitMinOp(ctx: MinOpContext): MergeOp = MergeOp.Min

  }

  object IterVisitor extends ArcBaseVisitor[Iter] {

    def visitChecked(tree: ParseTree): Iter =
      Option(this.visit(tree))
        .getOrElse(
          throw new AssertionError(s"Visiting a sub-tree returned null:\n${tree.toStringTree(parser)}")
        )

    def tokenToIterKind(t: Token): IterKind.IterKind =
      t.getType match {
        case ArcLexer.TScalarIter => IterKind.ScalarIter
        case ArcLexer.TSimdIter => IterKind.SimdIter
        case ArcLexer.TFringeIter => IterKind.FringeIter
        case ArcLexer.TNdIter => IterKind.NdIter
        case ArcLexer.TRangeIter => IterKind.RangeIter
        case ArcLexer.TKeyByIter => IterKind.KeyByIter
        case _ => IterKind.UnknownIter
      }

    override def visitSimpleIter(ctx: ArcParser.SimpleIterContext): Iter =
      Iter(
        kind = tokenToIterKind(ctx.iter),
        data = ExprVisitor.visitChecked(ctx.data)
      )

    override def visitFourIter(ctx: ArcParser.FourIterContext): Iter =
      Iter(
        kind = tokenToIterKind(ctx.iter),
        data = ExprVisitor.visitChecked(ctx.data),
        start = Some(ExprVisitor.visitChecked(ctx.start)),
        end = Some(ExprVisitor.visitChecked(ctx.end)),
        stride = Some(ExprVisitor.visitChecked(ctx.stride))
      )

    override def visitSixIter(ctx: ArcParser.SixIterContext): Iter =
      Iter(
        kind = IterKind.NdIter,
        data = ExprVisitor.visitChecked(ctx.data),
        start = Some(ExprVisitor.visitChecked(ctx.start)),
        end = Some(ExprVisitor.visitChecked(ctx.end)),
        stride = Some(ExprVisitor.visitChecked(ctx.stride)),
        shape = Some(ExprVisitor.visitChecked(ctx.shape)),
        strides = Some(ExprVisitor.visitChecked(ctx.strides))
      )

    override def visitRangeIter(ctx: ArcParser.RangeIterContext): Iter =
      Iter(
        kind = IterKind.RangeIter,
        data = Expr(
          kind = ExprKind.MakeVec(Vector.empty),
          ty = Type.Vec(Type.I64),
          ctx
        ),
        start = Some(ExprVisitor.visitChecked(ctx.start)),
        end = Some(ExprVisitor.visitChecked(ctx.end)),
        stride = Some(ExprVisitor.visitChecked(ctx.stride))
      )

    override def visitKeyByIter(ctx: ArcParser.KeyByIterContext): Iter =
      Iter(
        kind = IterKind.KeyByIter,
        data = ExprVisitor.visitChecked(ctx.data),
        keyFunc = Some(ExprVisitor.visitChecked(ctx.keyFunc))
      )

    override def visitUnkownIter(ctx: ArcParser.UnkownIterContext): Iter =
      Iter(
        kind = IterKind.UnknownIter,
        data = ExprVisitor.visitChecked(ctx.valueExpr())
      )
  }

  object TypeVisitor extends ArcBaseVisitor[Type] {

    def visitChecked(tree: ParseTree, allowNull: Boolean = false): Type = {
      if (allowNull && tree == null) {
        Type.unknown // just return a type variable
      } else {
        assert(tree != null, s"Can't extract type from null-tree")
        Option(this.visit(tree))
          .getOrElse {
            throw new AssertionError(s"Visiting a sub-tree returned null:\n${tree.toStringTree(parser)}")
          }
      }
    }

    def tokenToScalar(t: Token): Option[Type.Scalar] =
      Try(t.getType match {
        case ArcLexer.TI8 => Type.I8
        case ArcLexer.TI16 => Type.I16
        case ArcLexer.TI32 => Type.I32
        case ArcLexer.TI64 => Type.I64
        case ArcLexer.TU8 => Type.U8
        case ArcLexer.TU16 => Type.U16
        case ArcLexer.TU32 => Type.U32
        case ArcLexer.TU64 => Type.U64
        case ArcLexer.TF32 => Type.F32
        case ArcLexer.TF64 => Type.F64
        case ArcLexer.TBool => Type.Bool
        case ArcLexer.TUnit => Type.UnitT
      }).toOption

    override def visitI8(ctx: I8Context): Type = Type.I8

    override def visitI16(ctx: I16Context): Type = Type.I16

    override def visitI32(ctx: I32Context): Type = Type.I32

    override def visitI64(ctx: I64Context): Type = Type.I64

    override def visitU8(ctx: U8Context): Type = Type.U8

    override def visitU16(ctx: U16Context): Type = Type.U16

    override def visitU32(ctx: U32Context): Type = Type.U32

    override def visitU64(ctx: U64Context): Type = Type.U64

    override def visitF32(ctx: F32Context): Type = Type.F32

    override def visitF64(ctx: F64Context): Type = Type.F64

    override def visitBool(ctx: BoolContext): Type = Type.Bool

    override def visitUnitT(ctx: UnitTContext): Type = Type.UnitT

    override def visitStringT(ctx: StringTContext): Type = Type.StringT

    override def visitSimd(ctx: SimdContext): Type =
      Type.Simd(
        elemTy = this.visitChecked(ctx.elemT)
      )

    override def visitVec(ctx: VecContext): Type =
      Type.Vec(
        elemTy = this.visitChecked(ctx.elemT)
      )

    override def visitStream(ctx: StreamContext): Type =
      Type.Stream(
        elemTy = this.visitChecked(ctx.elemT)
      )

    override def visitDict(ctx: DictContext): Type =
      Type.Dict(
        keyTy = this.visitChecked(ctx.keyT),
        valueTy = this.visitChecked(ctx.valueT)
      )

    override def visitStruct(ctx: StructContext): Type =
      Type.Struct(
        elemTys = ctx.types.asScala
          .map(TypeVisitor.visitChecked(_))
          .toVector
      )

    override def visitUnitFunction(ctx: ArcParser.UnitFunctionContext): Type =
      Type.Function(
        params = Vector.empty,
        returnTy = this.visitChecked(ctx.returnT)
      )

    override def visitParamFunction(ctx: ArcParser.ParamFunctionContext): Type =
      Type.Function(
        params = ctx.paramTypes.asScala
          .map(this.visitChecked(_))
          .toVector,
        returnTy = this.visitChecked(ctx.returnT)
      )

    override def visitAppender(ctx: ArcParser.AppenderContext): Type =
      Type.Builder.Appender(
        elemTy = TypeVisitor.visitChecked(ctx.elemT),
        annotations = AnnotationVisitor.visitChecked(ctx.annotations())
      )

    override def visitStreamAppender(ctx: ArcParser.StreamAppenderContext): Type =
      Type.Builder.StreamAppender(
        elemTy = TypeVisitor.visitChecked(ctx.elemT),
        annotations = AnnotationVisitor.visitChecked(ctx.annotations())
      )

    override def visitMerger(ctx: ArcParser.MergerContext): Type =
      Type.Builder.Merger(
        elemTy = TypeVisitor.visitChecked(ctx.elemT),
        opTy = OpVisitor.visitChecked(ctx.commutativeBinop()),
        annotations = AnnotationVisitor.visitChecked(ctx.annotations())
      )

    override def visitDictMerger(ctx: ArcParser.DictMergerContext): Type =
      Type.Builder.DictMerger(
        keyTy = TypeVisitor.visitChecked(ctx.keyT),
        valueTy = TypeVisitor.visitChecked(ctx.valueT),
        opTy = OpVisitor.visitChecked(ctx.commutativeBinop()),
        annotations = AnnotationVisitor.visitChecked(ctx.annotations())
      )

    override def visitGroupMerger(ctx: ArcParser.GroupMergerContext): Type =
      Type.Builder.GroupMerger(
        keyTy = TypeVisitor.visitChecked(ctx.keyT),
        valueTy = TypeVisitor.visitChecked(ctx.valueT),
        annotations = AnnotationVisitor.visitChecked(ctx.annotations())
      )

    override def visitVecMerger(ctx: ArcParser.VecMergerContext): Type =
      Type.Builder.VecMerger(
        elemTy = TypeVisitor.visitChecked(ctx.elemT),
        opTy = OpVisitor.visitChecked(ctx.commutativeBinop()),
        annotations = AnnotationVisitor.visitChecked(ctx.annotations())
      )

    override def visitWindower(ctx: ArcParser.WindowerContext): Type =
      Type.Builder.Windower(
        discTy = TypeVisitor.visitChecked(ctx.discT),
        aggrTy = TypeVisitor.visitChecked(ctx.aggrT),
        aggrMergeTy = TypeVisitor.visitChecked(ctx.aggrMergeT),
        aggrResultTy = TypeVisitor.visitChecked(ctx.aggrResultT),
        annotations = AnnotationVisitor.visitChecked(ctx.annotations())
      )

    // ignore the annotated number to avoid clashes
    override def visitTypeVariable(ctx: ArcParser.TypeVariableContext): Type = Type.unknown
  }

}
