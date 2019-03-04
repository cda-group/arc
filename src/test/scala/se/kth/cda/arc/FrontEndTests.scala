package se.kth.cda.arc

import org.antlr.v4.runtime.{CharStreams, CommonTokenStream}
import org.scalatest.{Assertion, FunSuite, Matchers}
import se.kth.cda.arc.transform.MacroExpansion
import se.kth.cda.arc.typeinference.TypeInference

class FrontEndTests extends FunSuite with Matchers {

  implicit class StringTokenAux(val input: String) {
    def analyze: String = {
      val inputStream = CharStreams.fromString(input)
      val lexer = new ArcLexer(inputStream)
      val tokenStream = new CommonTokenStream(lexer)
      val parser = new ArcParser(tokenStream)
      val tree = parser.expr()
      val translator = ASTTranslator(parser)
      val ast = translator.translate(tree)
      val expanded = MacroExpansion.expand(ast).get
      val typed = TypeInference.solve(expanded).get
      PrettyPrint.print(typed)
    }
  }

  implicit class TestHelper(val result: String) {
    def shouldBeApprox(expected: String): Assertion = {
      val fmt_result = result.trim.replaceAll("\n|", "").replace(" ", "") // get out of whitespace-hell
      val fmt_expected = expected.trim.replace("\n", "").replace(" ", "")
      fmt_result shouldBe fmt_expected
    }
  }

  test("basic") {
    "let x: i32 = 5; let y = x; y".analyze shouldBeApprox
      "( let x:i32=5:i32; let y:i32=x:i32; y ):i32"
  }
  test("lookup vec") {
    "lookup([0,1,2], 1)"
  }
  test("lookup dict") {
    "lookup(result(for(iter([{1,2},{1,4}]), groupmerger[i32,i32], |b,i,y| merge(b,y))), 1)"
  }
  test("hash") {
    "hash(1,2,3)"
  }
  test("matrix multiplication") {
    """
    let n=2L;
    let p=2L;
    let m=3L;
    let A=[1,2,3,4,5,6];
    let B=[1,2,4,5,7,8];
    let C=for(rangeiter(0L,n,1L),
          appender[i32],
          |Cn,xn,i|
            for(rangeiter(0L,p,1L),
                Cn,
                |Cp,xp,j|
                  ( let s=for(rangeiter(0L,m,1L),
                        merger[i32,+],
                        |sum,xm,k|
                          ( let Aik=lookup(A,((i*n)+k));
                            let Bkj=lookup(B,((k*m)+j));
                            merge(sum,Aik*Bkj)
                          )
                    );
                    merge(Cp,result(s))
                  )
            )
      );
    result(C)
    """.analyze shouldBeApprox
    """
    ( let n:i64=2L:i64;
      let p:i64=2L:i64;
      let m:i64=3L:i64;
      let A:vec[i32]=[1,2,3,4,5,6]:vec[i32];
      let B:vec[i32]=[1,2,4,5,7,8]:vec[i32];
      let C:appender[i32]=for(rangeiter([]:vec[i64],0L:i64,n:i64,1L:i64),
            appender[i32],
            |Cn:appender[i32],xn:i64,i:i64|
              for(rangeiter([]:vec[i64],0L:i64,p:i64,1L:i64),
                  Cn,
                  |Cp:appender[i32],xp:i64,j:i64|
                    ( let s:merger[i32,+]=for(rangeiter([]:vec[i64],0L:i64,m:i64,1L:i64),
                          merger[i32,+],
                          |sum:merger[i32,+],xm:i64,k:i64|
                            ( let Aik:i32=lookup(A:vec[i32],((i:i64*n:i64):i64+k:i64):i64):i32;
                              let Bkj:i32=lookup(B:vec[i32],((k:i64*m:i64):i64+j:i64):i64):i32;
                              merge(sum,Aik:i32*Bkj:i32)
                            ):merger[i32,+]
                      ):merger[i32,+];
                      merge(Cp,result(s))
                    ):appender[i32]
              ):appender[i32]
        ):appender[i32];
      result(C)
    ):vec[i32]
    """
  }
  test("pagerank") {
    """
    let src = [0L,0L,1L,2L];
    let dst = [1L,2L,2L,0L];

    let out_edges = zip(src, dst);
    let in_edges  = zip(dst, src);

    let in_nbrs = result(for(iter(in_edges), groupmerger[i64,i64], |b,i,x| merge(b, x)));

    let fan_outs = result(for(iter(out_edges), dictmerger[i64,i64,+], |b,i,x| merge(b, {x.$0, 1L})));

    let n = result(for(iter(zip(src, dst)), merger[i64,max], |b,i,x| merge(merge(b, x.$0), x.$1))) + 1L;

    let initial_rank = 1.0/f64(n);
    let initial_ranks = result(for(rangeiter(0L, n, 1L), appender, |b,i,x| merge(b, initial_rank)));

    let start = {initial_ranks, 0};

    let teleport       = 0.1;
    let tolerance      = 0.0001;
    let max_iterations = 20;

    iterate(start, |iterator|
      let old_ranks = iterator.$0;
      let iteration = iterator.$1;

      let new_ranks = result(for(rangeiter(0L, n, 1L), appender, |ranks,i,node|
        let in_nbrs_of_node = lookup(in_nbrs, node);
        let s = result(for(iter(in_nbrs_of_node), merger[f64,+], |sum,j,in_nbr|
          let fan_out = lookup(fan_outs, in_nbr);
          let old_rank = lookup(old_ranks, in_nbr);
          merge(sum, old_rank / f64(fan_out))
        ));
        merge(ranks, teleport/f64(n) + (1.0 - teleport) * s)
      ));

      let max_delta = result(for(iter(zip(new_ranks,old_ranks)), merger[f64,max], |b,i,x|
        let delta = x.$0-x.$1;
        let abs = select(delta < 0.0, -1.0*delta, delta);
        merge(b, abs)
      ));
      let cond1 = select(iteration < max_iterations, true, false);
      let cond2 = select(max_delta < tolerance, false, true);
      {{new_ranks, iteration+1}, cond1 && cond2}
    )
    """.analyze shouldBeApprox
    """
    ( let src:vec[i64]=[0L,0L,1L,2L]:vec[i64];
    let dst:vec[i64]=[1L,2L,2L,0L]:vec[i64];
    let out_edges:vec[{i64,i64}]=zip(src,dst):vec[{i64,i64}];
    let in_edges:vec[{i64,i64}]=zip(dst,src):vec[{i64,i64}];
    let in_nbrs:dict[i64, vec[i64]]=result(for(in_edges:vec[{i64,i64}],
            groupmerger[i64,i64],
            |b:groupmerger[i64,i64],i:i64,x:{i64,i64}|
              merge(b,x):groupmerger[i64,i64]
        )):dict[i64, vec[i64]];
    let fan_outs:dict[i64, i64]=result(for(out_edges:vec[{i64,i64}],
            merger[i64,i64,+],
            |b:merger[i64,i64,+],i:i64,x:{i64,i64}|
              merge(b,{x.$0,1L}):merger[i64,i64,+]
        )):dict[i64, i64];
    let n:i64=(result(for(zip(src,dst):vec[{i64,i64}],
             merger[i64,max],
             |b:merger[i64,max],i:i64,x:{i64,i64}|
               merge(merge(b,x.$0),x.$1):merger[i64,max]
         )):i64+1L:i64):i64;
    let initial_rank:f64=(1.0:f64/f64(n)):f64;
    let initial_ranks:vec[f64]=result(for(rangeiter([]:vec[i64],0L:i64,n:i64,1L:i64),
            appender[f64],
            |b:appender[f64],i:i64,x:i64|
              merge(b,initial_rank):appender[f64]
        )):vec[f64];
    let start:{vec[f64],i32}={initial_ranks,0}:{vec[f64],i32};
    let teleport:f64=0.1:f64;
    let tolerance:f64=0.0001:f64;
    let max_iterations:i32=20:i32;
    iterate (start:{vec[f64],i32},
      |iterator:{vec[f64],i32}|
        ( let old_ranks:vec[f64]=(iterator.$0):vec[f64];
          let iteration:i32=(iterator.$1):i32;
          let new_ranks:vec[f64]=result(for(rangeiter([]:vec[i64],0L:i64,n:i64,1L:i64),
                  appender[f64],
                  |ranks:appender[f64],i:i64,node:i64|
                    ( let in_nbrs_of_node:vec[i64]=lookup(in_nbrs:dict[i64, vec[i64]],node:i64):vec[i64];
                      let s:f64=result(for(in_nbrs_of_node:vec[i64],
                              merger[f64,+],
                              |sum:merger[f64,+],j:i64,in_nbr:i64|
                                ( let fan_out:i64=lookup(fan_outs:dict[i64, i64],in_nbr:i64):i64;
                                  let old_rank:f64=lookup(old_ranks:vec[f64],in_nbr:i64):f64;
                                  merge(sum,old_rank:f64/f64(fan_out))
                                ):merger[f64,+]
                          )):f64;
                      merge(ranks,(teleport:f64/f64(n)):f64+((1.0:f64-teleport:f64):f64*s:f64):f64)
                    ):appender[f64]
              )):vec[f64];
          let max_delta:f64=result(for(zip(new_ranks,old_ranks):vec[{f64,f64}],
                  merger[f64,max],
                  |b:merger[f64,max],i:i64,x:{f64,f64}|
                    ( let delta:f64=((x.$0):f64-(x.$1):f64):f64;
                      let abs:f64=select(delta:f64<0.0:f64,
                          -(1.0:f64*delta:f64):f64,
                          delta:f64
                        ):f64;
                      merge(b,abs)
                    ):merger[f64,max]
              )):f64;
          let cond1:bool=select(iteration:i32<max_iterations:i32,
              true:bool,
              false:bool
            ):bool;
          let cond2:bool=select(max_delta:f64<tolerance:f64,
              false:bool,
              true:bool
            ):bool;
          {{new_ranks,iteration:i32+1:i32},cond1:bool&&cond2:bool}
          ):{{vec[f64],i32},bool}
      )
    ):{vec[f64],i32}
    """
  }
}