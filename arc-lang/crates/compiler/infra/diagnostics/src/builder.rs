use std::borrow::Cow;

use info::Info;

pub struct DiagnosticAt {
    text: Cow<'static, str>,
    info: Info,
}

pub struct DiagnosticAtAnd {
    text0: Cow<'static, str>,
    info0: Info,
    text1: Cow<'static, str>,
    info1: Info,
}

pub trait DiagnoseAt {
    fn at(self, info: Info) -> DiagnosticAt;
}

impl DiagnoseAt for &'static str {
    fn at(self, info: Info) -> DiagnosticAt {
        DiagnosticAt {
            text: self.into(),
            info,
        }
    }
}

impl DiagnoseAt for String {
    fn at(self, info: Info) -> DiagnosticAt {
        DiagnosticAt {
            text: self.into(),
            info,
        }
    }
}

trait DiagnoseAtAnd {
    fn and(self, info: Info) -> DiagnosticAtAnd;
}

impl DiagnosticAt {
    pub fn and(self, text1: &'static str, info1: Info) -> DiagnosticAtAnd {
        DiagnosticAtAnd {
            text0: self.text,
            info0: self.info,
            text1: text1.into(),
            info1: self.info,
        }
    }
}
