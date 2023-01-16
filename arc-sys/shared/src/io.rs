use tokio::process::ChildStdin;
use tokio::process::ChildStdout;
use tokio_serde::formats::Json;
use tokio_serde::Framed;
use tokio_util::codec::FramedRead;
use tokio_util::codec::FramedWrite;
use tokio_util::codec::LengthDelimitedCodec;

pub type Receiver<I> = Framed<FramedRead<ChildStdout, LengthDelimitedCodec>, I, I, Json<I, I>>;
pub type Sender<O> = Framed<FramedWrite<ChildStdin, LengthDelimitedCodec>, O, O, Json<O, O>>;
