use crate::*;

use std::hash::Hash;

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.kind.as_ref() {
            VAggregator(a) => unreachable!(),
            VArray(a) => a.hash(state),
            VBlob(a) => unreachable!(),
            VBool(a) => a.hash(state),
            VChar(a) => a.hash(state),
            VDiscretizer(a) => unreachable!(),
            VDuration(a) => a.hash(state),
            VEncoding(a) => unreachable!(),
            VF64(a) => unreachable!(),
            VFile(a) => unreachable!(),
            VFunction(a) => unreachable!(),
            VI32(a) => a.hash(state),
            VMatrix(a) => unreachable!(),
            VModel(a) => unreachable!(),
            VOption(a) => a.hash(state),
            VPath(a) => a.hash(state),
            VReader(a) => unreachable!(),
            VRecord(a) => a.hash(state),
            VVariant(a) => a.hash(state),
            VResult(a) => a.hash(state),
            VSocketAddr(a) => unreachable!(),
            VStream(a) => unreachable!(),
            VString(a) => a.hash(state),
            VTime(a) => a.hash(state),
            VTimeSource(a) => unreachable!(),
            VTuple(a) => a.hash(state),
            VUsize(a) => a.hash(state),
            VUnit(a) => a.hash(state),
            VUrl(a) => a.hash(state),
            VVec(a) => a.hash(state),
            VWriter(a) => unreachable!(),
            _ => unreachable!(),
        }
    }
}
