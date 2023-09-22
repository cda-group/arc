use std::fmt::Write;
use std::io::Read;

use minus::ExitStrategy;

use crate::blob::Blob;
use crate::path::Path;
use crate::string::String;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct File(pub std::rc::Rc<std::cell::RefCell<std::fs::File>>);

impl File {
    pub fn open(path: impl Into<Path>) -> Self {
        File::from(std::fs::File::open(&path.into().0).unwrap())
    }

    pub fn read_to_string(self) -> String {
        let mut string = std::string::String::new();
        self.0.borrow_mut().read_to_string(&mut string).unwrap();
        String::from(string)
    }

    pub fn read_to_bytes(self) -> Blob {
        let mut vec = std::vec::Vec::new();
        self.0.borrow_mut().read_to_end(&mut vec).unwrap();
        Blob::new(vec)
    }

    pub fn inspect(self) {
        let mut string = std::string::String::new();
        self.0.borrow_mut().read_to_string(&mut string).unwrap();
        if let Err(e) = logs(string) {
            eprintln!("{}", e);
        }
        fn logs(string: std::string::String) -> Result<(), std::fmt::Error> {
            let mut pager = minus::Pager::new();

            for line in string.lines() {
                writeln!(pager, "{}", line).unwrap();
            }
            pager.set_exit_strategy(ExitStrategy::PagerQuit).unwrap();
            minus::page_all(pager).unwrap();
            Ok(())
        }
        ().into()
    }
}

impl From<std::fs::File> for File {
    fn from(file: std::fs::File) -> Self {
        Self(std::rc::Rc::new(std::cell::RefCell::new(file)))
    }
}
