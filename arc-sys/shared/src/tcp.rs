use serde::de::DeserializeOwned;
use tokio::net::tcp::OwnedReadHalf;
use tokio::net::tcp::OwnedWriteHalf;
use tokio_serde::formats::Json;
use tokio_serde::Framed;
use tokio_util::codec::FramedRead;
use tokio_util::codec::FramedWrite;
use tokio_util::codec::LengthDelimitedCodec;

pub type Receiver<I> = Framed<FramedRead<OwnedReadHalf, LengthDelimitedCodec>, I, I, Json<I, I>>;
pub type Sender<O> = Framed<FramedWrite<OwnedWriteHalf, LengthDelimitedCodec>, O, O, Json<O, O>>;
