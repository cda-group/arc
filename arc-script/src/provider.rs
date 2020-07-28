use {
    crate::{ast::*, error::Error},
    //     http::uri::Uri,
    pb_rs::types::*,
    reqwest::blocking,
    std::{fs::File, io::copy, path::Path},
    tempfile::Builder,
};

impl Expr {
    pub fn download(&mut self) {}
}

#[allow(unused)]
fn parse_uri(url: &str) -> Result<(), Error> {
    let tmp_dir = Builder::new().prefix("tmp").tempdir()?;
    let text = blocking::get(url)?.text()?;
    let path = tmp_dir.path().join(Path::new("tmp.proto"));
    let mut dest = File::create(&path)?;
    copy(&mut text.as_bytes(), &mut dest)?;
    if let Ok(pb) = FileDescriptor::read_proto(&path, &[]) {
        println!("{:?}", pb.import_paths);
        println!("{:?}", pb.package);
        println!("{:?}", pb.syntax);
        println!("{:?}", pb.messages);
        println!("{:?}", pb.enums);
        println!("{:?}", pb.module);
        println!("{:?}", pb.rpc_services);
        println!("{:?}", pb.owned);
        Ok(())
    } else {
        Ok(())
    }
}

// pub fn _parse_uri<'i, 'e>(
//     Spanned(l, s, r): Spanned<&'i str>,
//     errors: &'e mut Vec<ErrorRecovery>,
// ) -> Option<Uri> {
//     let unquoted = &s[1..s.len() - 1];
//     match Uri::from_str(unquoted) {
//         Err(error) => {
//             let span = Span::new(l as u32, r as u32);
//             let msg = error.to_string();
//             let error = CompilerError::BadUri { msg, span }.into();
//             errors.push(error);
//             None
//         }
//         Ok(uri) => Some(uri),
//     }
// }
