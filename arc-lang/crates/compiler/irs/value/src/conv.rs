use builtins::aggregator::Aggregator;
use builtins::blob::Blob;
use builtins::dict::Dict;
use builtins::discretizer::Discretizer;
use builtins::duration::Duration;
use builtins::encoding::Encoding;
use builtins::file::File;
use builtins::model::Model;
use builtins::path::Path;
use builtins::reader::Reader;
use builtins::set::Set;
use builtins::socket::SocketAddr;
use builtins::time::Time;
use builtins::time_source::TimeSource;
use builtins::url::Url;
use builtins::writer::Writer;

use crate::dynamic::Array;
use crate::dynamic::Dataflow;
use crate::dynamic::Function;
use crate::dynamic::Instance;
use crate::dynamic::Matrix;
use crate::dynamic::Record;
use crate::dynamic::Stream;
use crate::dynamic::Tuple;
use crate::dynamic::Variant;
use crate::Value;
use crate::ValueKind;

macro_rules! conv {
    {
        $type:ty, $variant:ident, $as:ident
    } => {
        impl Value {
            #[track_caller]
            pub fn $as(&self) -> $type {
                if let ValueKind::$variant(v) = &*self.kind {
                    v.clone()
                } else {
                    unreachable!("{}{:?}", std::panic::Location::caller(), self);
                }
            }
        }
        impl From<$type> for Value {
            fn from(v: $type) -> Self {
                Value::new(ValueKind::$variant(v))
            }
        }
    }
}

conv!((), VUnit, as_unit);
conv!(Array, VArray, as_array);
conv!(Tuple, VTuple, as_tuple);
conv!(Function, VFunction, as_function);
conv!(Matrix, VMatrix, as_matrix);
conv!(Record, VRecord, as_record);
conv!(Stream, VStream, as_stream);
conv!(Variant, VVariant, as_variant);
conv!(bool, VBool, as_bool);
conv!(Aggregator<Function, Function, Function, Function>, VAggregator, as_aggregator);
conv!(Blob, VBlob, as_blob);
conv!(Dict<Value, Value>, VDict, as_dict);
conv!(Discretizer, VDiscretizer, as_discretizer);
conv!(Duration, VDuration, as_duration);
conv!(Dataflow, VDataflow, as_dataflow);
conv!(Encoding, VEncoding, as_encoding);
conv!(File, VFile, as_file);
conv!(Model, VModel, as_model);
conv!(builtins::option::Option<Value>, VOption, as_option);
conv!(Path, VPath, as_path);
conv!(Reader, VReader, as_reader);
conv!(builtins::result::Result<Value>, VResult, as_result);
conv!(Set<Value>, VSet, as_set);
conv!(SocketAddr, VSocketAddr, as_socket_addr);
conv!(builtins::string::String, VString, as_string);
conv!(Time, VTime, as_time);
conv!(TimeSource<Function>, VTimeSource, as_time_source);
conv!(Url, VUrl, as_url);
conv!(builtins::vec::Vec<Value>, VVec, as_vec);
conv!(Writer, VWriter, as_writer);
conv!(char, VChar, as_char);
conv!(f32, VF32, as_f32);
conv!(f64, VF64, as_f64);
conv!(i128, VI128, as_i128);
conv!(i16, VI16, as_i16);
conv!(i32, VI32, as_i32);
conv!(i64, VI64, as_i64);
conv!(i8, VI8, as_i8);
conv!(u128, VU128, as_u128);
conv!(u16, VU16, as_u16);
conv!(u32, VU32, as_u32);
conv!(u64, VU64, as_u64);
conv!(u8, VU8, as_u8);
conv!(usize, VUsize, as_usize);
conv!(Instance, VInstance, as_instance);
