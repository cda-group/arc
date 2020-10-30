use {
    anyhow::Result,
    std::{fs::File, io::prelude::*, path::Path},
};

pub fn read_file<P: AsRef<Path>>(path: &P) -> Result<String> {
    let mut file = File::open(path)?;
    let mut source = String::new();
    file.read_to_string(&mut source)?;
    Ok(source)
}
