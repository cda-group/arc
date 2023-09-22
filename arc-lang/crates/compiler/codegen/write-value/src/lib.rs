#![allow(unused)]
use builtins::aggregator::Aggregator;
use builtins::discretizer::Discretizer;
use builtins::encoding::Encoding;
use builtins::model::Model;
use builtins::path::Path;
use builtins::reader::Reader;
use builtins::time_source::TimeSource;
use builtins::writer::Writer;
use codegen::Context;
use colors::BUILTIN_COLOR;
use colors::NUMERIC_COLOR;
use colors::STRING_COLOR;
use std::io::Result;
use std::io::Write;
use value::dynamic::Dataflow;
use value::dynamic::Function;
use value::dynamic::Sink;
use value::dynamic::Stream;
use value::dynamic::StreamKind;

use hir::*;
use value::*;

pub fn write_value_type(ctx: &mut Context<impl Write>, (v, t): &(Value, Type)) -> Result<()> {
    write_value(ctx, v)?;
    ctx.colon()?;
    write_hir::write_type(ctx, t)
}

pub fn write_value(ctx: &mut Context<impl Write>, v: &Value) -> Result<()> {
    match v.kind.as_ref() {
        VBool(v) => {
            ctx.bold_colored(&format!("{}", v), BUILTIN_COLOR)?;
        }
        VString(v) => {
            ctx.colored("\"", STRING_COLOR)?;
            ctx.colored(&v.as_ref(), STRING_COLOR)?;
            ctx.colored("\"", STRING_COLOR)?;
        }
        VDuration(v) => {
            ctx.fmt(format_args!("{:?}", v.0))?;
        }
        VI32(v) => {
            ctx.colored(&format!("{}", &v), NUMERIC_COLOR)?;
        }
        VF32(v) => {
            ctx.colored(&format!("{}", &v), NUMERIC_COLOR)?;
        }
        VUnit(()) => {
            ctx.lit("()")?;
        }
        VChar(v) => {
            ctx.colored(&format!("'{}'", v), STRING_COLOR)?;
        }
        VFunction(v) => {
            write_function(ctx, v)?;
        }
        VRecord(v) => {
            ctx.brace(|ctx| {
                ctx.seq(&v.0, |ctx, (x, v)| {
                    ctx.lit(&x)?;
                    ctx.colon()?;
                    write_value(ctx, v)
                })
            })?;
        }
        VVariant(v) => {
            write_value(ctx, &v.v)?;
            ctx.lit(" as ")?;
            ctx.lit(&v.x)?;
        }
        VTime(v) => {
            ctx.fmt(format_args!("{:?}", v.0))?;
        }
        VStream(v) => {
            ctx.lit("Stream(..)")?;
        }
        VDiscretizer(v) => {
            write_discretizer(ctx, v)?;
        }
        VAggregator(v) => {
            write_aggregator(ctx, v)?;
        }
        VReader(v) => {
            write_reader(ctx, v)?;
        }
        VWriter(v) => {
            write_writer(ctx, v)?;
        }
        VEncoding(v) => {
            write_encoding(ctx, v)?;
        }
        VTimeSource(v) => {
            write_time_source(ctx, v)?;
        }
        VModel(v) => {
            write_model(ctx, v)?;
        }
        VArray(v) => {
            ctx.brack(|ctx| ctx.indented_seq(&v.0, write_value))?;
        }
        VTuple(v) => {
            ctx.paren(|ctx| ctx.indented_seq(&v.0, write_value))?;
        }
        VResult(v) => {
            write_result(ctx, v)?;
        }
        VPath(v) => {
            write_path(ctx, v)?;
        }
        VFile(v) => {
            ctx.fmt(format_args!("{:?}", v.0.borrow().metadata().unwrap()))?;
        }
        VBlob(v) => {
            ctx.fmt(format_args!("{:?}", v.0))?;
        }
        VDict(v) => {
            ctx.brace(|ctx| {
                ctx.indented_seq(v.0.iter(), |ctx, (k, v)| {
                    write_value(ctx, k)?;
                    ctx.colon()?;
                    write_value(ctx, v)
                })
            })?;
        }
        VF64(v) => {
            ctx.colored(&format!("{}", &v), NUMERIC_COLOR)?;
        }
        VI128(v) => {
            ctx.colored(&format!("{}", &v), NUMERIC_COLOR)?;
        }
        VI16(v) => {
            ctx.colored(&format!("{}", &v), NUMERIC_COLOR)?;
        }
        VI64(v) => {
            ctx.colored(&format!("{}", &v), NUMERIC_COLOR)?;
        }
        VI8(v) => {
            ctx.colored(&format!("{}", &v), NUMERIC_COLOR)?;
        }
        VMatrix(v) => {
            ctx.lit(&format!("{:?}", v))?;
        }
        VOption(v) => todo!(),
        VSet(v) => todo!(),
        VSocketAddr(v) => todo!(),
        VDataflow(v) => {
            ctx.lit("Dataflow(...)")?;
        }
        VU128(v) => todo!(),
        VU16(v) => todo!(),
        VU32(v) => todo!(),
        VU64(v) => todo!(),
        VU8(v) => todo!(),
        VUrl(v) => todo!(),
        VUsize(v) => todo!(),
        VVec(v) => todo!(),
        VInstance(v) => {
            ctx.lit("Instance(...)")?;
        }
    }
    Ok(())
}

pub fn write_writer(ctx: &mut Context<impl Write>, d: &Writer) -> Result<()> {
    match d {
        Writer::Stdout => {
            ctx.def("stdout")?;
            ctx.paren(|_| Ok(()))?;
        }
        Writer::File { path } => {
            ctx.def("file")?;
            ctx.paren(|ctx| ctx.text(&format!("{:?}", path.0.display())))?;
        }
        Writer::Http { url } => {
            ctx.def("http")?;
            ctx.paren(|ctx| ctx.text(&format!("{}", url.0)))?;
        }
        Writer::Tcp { addr: ip } => {
            ctx.def("tcp")?;
            ctx.paren(|ctx| ctx.text(&format!("{}", ip.0)))?;
        }
        Writer::Kafka {
            addr: broker,
            topic,
        } => {
            ctx.def("kafka")?;
            ctx.paren(|ctx| {
                ctx.text(&format!("{}", broker.0))?;
                ctx.comma()?;
                ctx.text(&format!("{}", topic.as_ref()))
            })?;
        }
    }
    Ok(())
}

pub fn write_reader(ctx: &mut Context<impl Write>, d: &Reader) -> Result<()> {
    match d {
        Reader::Stdin => {
            ctx.def("stdin")?;
            ctx.paren(|_| Ok(()))?;
        }
        Reader::File { path, watch } => {
            ctx.def("file")?;
            ctx.paren(|ctx| ctx.text(&format!("{:?}", path.0.display())))?;
        }
        Reader::Http { url } => {
            ctx.def("http")?;
            ctx.paren(|ctx| ctx.text(&format!("{}", url.0)))?;
        }
        Reader::Tcp { addr: ip } => {
            ctx.def("tcp")?;
            ctx.paren(|ctx| ctx.text(&format!("{}", ip.0)))?;
        }
        Reader::Kafka {
            addr: broker,
            topic,
        } => {
            ctx.def("kafka")?;
            ctx.paren(|ctx| {
                ctx.text(&format!("{}", broker.0))?;
                ctx.comma()?;
                ctx.text(&format!("{}", topic.as_ref()))
            })?;
        }
    }
    Ok(())
}

pub fn write_dataflow(ctx: &mut Context<impl Write>, d: &Dataflow) -> Result<()> {
    ctx.lit("Dataflow")?;
    ctx.brace(|ctx| {
        ctx.seq(&d.streams, write_stream)?;
        ctx.seq(&d.sinks, write_sink)
    })
}

pub fn write_sink(ctx: &mut Context<impl Write>, s: &Sink) -> Result<()> {
    let (x, w, e) = s.0.as_ref();
    ctx.def("sink")?;
    ctx.paren(|ctx| {
        ctx.val(&x)?;
        ctx.comma()?;
        write_writer(ctx, w)?;
        ctx.comma()?;
        write_encoding(ctx, e)
    })
}

pub fn write_stream(ctx: &mut Context<impl Write>, s: &Stream) -> Result<()> {
    fn write_stream(ctx: &mut Context<impl Write>, s: &Stream) -> Result<()> {
        match s.kind.as_ref() {
            StreamKind::DSource(r, e, f) => {
                ctx.def("source")?;
                ctx.paren(|ctx| {
                    write_reader(ctx, r)?;
                    ctx.comma()?;
                    write_encoding(ctx, e)?;
                    ctx.comma()?;
                    write_time_source(ctx, f)
                })?;
            }
            StreamKind::DMap(x, f) => {
                ctx.def("map")?;
                ctx.paren(|ctx| {
                    ctx.val(&x)?;
                    ctx.comma()?;
                    write_function(ctx, f)
                })?;
            }
            StreamKind::DFilter(x, f) => {
                ctx.def("filter")?;
                ctx.paren(|ctx| {
                    ctx.val(&x)?;
                    ctx.comma()?;
                    write_function(ctx, f)
                })?;
            }
            StreamKind::DFlatten(x) => {
                ctx.def("flatten")?;
                ctx.paren(|ctx| ctx.val(&x))?;
            }
            StreamKind::DFlatMap(x, f) => {
                ctx.def("flat_map")?;
                ctx.paren(|ctx| {
                    ctx.val(&x)?;
                    ctx.comma()?;
                    write_function(ctx, f)
                })?;
            }
            StreamKind::DScan(x, f) => {
                ctx.def("scan")?;
                ctx.paren(|ctx| {
                    ctx.val(&x)?;
                    ctx.comma()?;
                    write_function(ctx, f)
                })?;
            }
            StreamKind::DKeyby(x, f) => {
                ctx.def("keyby")?;
                ctx.paren(|ctx| {
                    ctx.val(&x)?;
                    ctx.comma()?;
                    write_function(ctx, f)
                })?;
            }
            StreamKind::DUnkey(x) => {
                ctx.def("unkey")?;
                ctx.paren(|ctx| ctx.val(&x))?;
            }
            StreamKind::DApply(x, f) => {
                ctx.def("apply")?;
                ctx.paren(|ctx| {
                    ctx.val(&x)?;
                    ctx.comma()?;
                    write_function(ctx, f)
                })?;
            }
            StreamKind::DWindow(x, d, a) => {
                ctx.def("window")?;
                ctx.paren(|ctx| {
                    ctx.val(&x)?;
                    ctx.comma()?;
                    write_discretizer(ctx, d)?;
                    ctx.comma()?;
                    write_aggregator(ctx, a)
                })?;
            }
            StreamKind::DMerge(xs) => {
                ctx.def("merge")?;
                ctx.paren(|ctx| ctx.seq(xs, |ctx, x| ctx.val(&x)))?;
            }
        }
        Ok(())
    }
    ctx.keyword("val ")?;
    ctx.val(&s.name)?;
    ctx.lit(" = ")?;
    write_stream(ctx, s)?;
    ctx.lit(";")
}

fn write_function(ctx: &mut Context<impl Write>, f: &Function) -> Result<()> {
    ctx.def(&f.0)
}

fn write_encoding(ctx: &mut Context<impl Write>, e: &Encoding) -> Result<()> {
    match e {
        Encoding::Csv { sep } => {
            ctx.def("csv")?;
            ctx.paren(|ctx| ctx.fmt(format_args!("{sep}")))
        }
        Encoding::Json => {
            ctx.def("json")?;
            ctx.paren(|_| Ok(()))
        }
    }
}

fn write_time_source(ctx: &mut Context<impl Write>, e: &TimeSource<Function>) -> Result<()> {
    match e {
        TimeSource::Ingestion { watermark_interval } => {
            ctx.def("ingestion")?;
            ctx.paren(|ctx| ctx.numeric(&format!("{:?}", watermark_interval)))
        }
        TimeSource::Event {
            extractor,
            watermark_interval,
            slack,
        } => {
            ctx.def("event")?;
            ctx.paren(|ctx| {
                write_function(ctx, extractor)?;
                ctx.comma()?;
                ctx.numeric(&format!("{:?}", watermark_interval))?;
                ctx.comma()?;
                ctx.numeric(&format!("{:?}", slack))
            })
        }
    }
}

fn write_discretizer(ctx: &mut Context<impl Write>, d: &Discretizer) -> Result<()> {
    match d {
        Discretizer::Tumbling { length } => {
            ctx.def("tumbling")?;
            ctx.paren(|ctx| ctx.numeric(&format!("{:?}", length)))
        }
        Discretizer::Sliding { length, step } => {
            ctx.def("sliding")?;
            ctx.paren(|ctx| {
                ctx.numeric(&format!("{:?}", length))?;
                ctx.comma()?;
                ctx.numeric(&format!("{:?}", step))
            })
        }
        Discretizer::Session { gap } => {
            ctx.def("session")?;
            ctx.paren(|ctx| ctx.numeric(&format!("{:?}", gap)))
        }
        Discretizer::Counting { length } => {
            ctx.def("counting")?;
            ctx.paren(|ctx| ctx.numeric(&format!("{}", length)))
        }
        Discretizer::Moving { length, step } => {
            ctx.def("moving")?;
            ctx.paren(|ctx| {
                ctx.numeric(&format!("{}", length))?;
                ctx.comma()?;
                ctx.numeric(&format!("{}", step))
            })
        }
    }
}

fn write_aggregator(
    ctx: &mut Context<impl Write>,
    a: &Aggregator<Function, Function, Function, Function>,
) -> Result<()> {
    match a {
        Aggregator::Monoid {
            lift,
            combine,
            identity,
            lower,
        } => {
            ctx.def("Aggregator::monoid")?;
            ctx.paren(|ctx| {
                write_function(ctx, lift)?;
                ctx.comma()?;
                write_function(ctx, combine)?;
                ctx.comma()?;
                write_function(ctx, identity)?;
                ctx.comma()?;
                write_function(ctx, lower)
            })
        } // Aggregator::Compose { a0, a1 } => {
          //     ctx.def("Aggregator::compose")?;
          //     ctx.paren(|ctx| {
          //         write_aggregator(ctx, a0)?;
          //         ctx.comma()?;
          //         write_aggregator(ctx, a1)
          //     })
          // }
    }
}

fn write_result(ctx: &mut Context<impl Write>, v: &builtins::result::Result<Value>) -> Result<()> {
    match &v.0 {
        Ok(x) => {
            ctx.def("ok")?;
            ctx.paren(|ctx| write_value(ctx, x))
        }
        Err(x) => {
            ctx.def("err")?;
            ctx.paren(|ctx| ctx.fmt(format_args!("{x}")))
        }
    }
}

fn write_path(ctx: &mut Context<impl Write>, v: &Path) -> Result<()> {
    ctx.def("path")?;
    ctx.paren(|ctx| ctx.fmt(format_args!("{:?}", v.0.display())))
}

fn write_model(ctx: &mut Context<impl Write>, a: &Model) -> Result<()> {
    todo!()
}
