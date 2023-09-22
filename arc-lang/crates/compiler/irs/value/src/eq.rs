use crate::*;

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self.kind.as_ref(), other.kind.as_ref()) {
            (VAggregator(a), VAggregator(b)) => unreachable!(),
            (VArray(a), VArray(b)) => a == b,
            (VBlob(a), VBlob(b)) => unreachable!(),
            (VBool(a), VBool(b)) => a == b,
            (VChar(a), VChar(b)) => a == b,
            (VDiscretizer(a), VDiscretizer(b)) => unreachable!(),
            (VDuration(a), VDuration(b)) => a == b,
            (VEncoding(a), VEncoding(b)) => a == b,
            (VF32(a), VF32(b)) => unreachable!(),
            (VF64(a), VF64(b)) => unreachable!(),
            (VFile(a), VFile(b)) => unreachable!(),
            (VFunction(a), VFunction(b)) => a == b,
            (VI8(a), VI8(b)) => a == b,
            (VI16(a), VI16(b)) => a == b,
            (VI32(a), VI32(b)) => a == b,
            (VI64(a), VI64(b)) => a == b,
            (VU8(a), VU8(b)) => a == b,
            (VU16(a), VU16(b)) => a == b,
            (VU32(a), VU32(b)) => a == b,
            (VU64(a), VU64(b)) => a == b,
            (VMatrix(a), VMatrix(b)) => match (a, b) {
                (Matrix::I8(a), Matrix::I8(b)) => a == b,
                (Matrix::I16(a), Matrix::I16(b)) => a == b,
                (Matrix::I32(a), Matrix::I32(b)) => a == b,
                (Matrix::I64(a), Matrix::I64(b)) => a == b,
                (Matrix::U8(a), Matrix::U8(b)) => a == b,
                (Matrix::U16(a), Matrix::U16(b)) => a == b,
                (Matrix::U32(a), Matrix::U32(b)) => a == b,
                (Matrix::U64(a), Matrix::U64(b)) => a == b,
                (Matrix::F32(_), Matrix::F32(_)) => unreachable!(),
                (Matrix::F64(_), Matrix::F64(_)) => unreachable!(),
                _ => unreachable!(),
            },
            (VModel(a), VModel(b)) => unreachable!(),
            (VOption(a), VOption(b)) => a == b,
            (VPath(a), VPath(b)) => a == b,
            (VReader(a), VReader(b)) => unreachable!(),
            (VRecord(a), VRecord(b)) => a == b,
            (VVariant(a), VVariant(b)) => a == b,
            (VResult(a), VResult(b)) => a == b,
            (VSocketAddr(a), VSocketAddr(b)) => a == b,
            (VStream(a), VStream(b)) => unreachable!(),
            (VString(a), VString(b)) => a == b,
            (VTime(a), VTime(b)) => a == b,
            (VTimeSource(a), VTimeSource(b)) => unreachable!(),
            (VTuple(a), VTuple(b)) => a == b,
            (VUsize(a), VUsize(b)) => a == b,
            (VUnit(a), VUnit(b)) => a == b,
            (VUrl(a), VUrl(b)) => a == b,
            (VVec(a), VVec(b)) => a == b,
            (VWriter(a), VWriter(b)) => unreachable!(),
            (VDict(a), VDict(b)) => a == b,
            (VSet(a), VSet(b)) => a == b,
            _ => unreachable!(
                "Attempted to compare incompatible types \n{:?} \nand \n{:?}",
                self, other
            ),
        }
    }
}

impl Eq for Value {}
