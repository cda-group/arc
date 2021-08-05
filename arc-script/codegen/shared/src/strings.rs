use derive_more::Deref;
use derive_more::From;

#[derive(From, Deref)]
#[from(forward)]
pub struct String {
    pub concrete: std::rc::Rc<std::string::String>,
}

mod send {
    use crate::conversions::Convert;
    use derive_more::Deref;
    use derive_more::From;

    #[derive(From, Deref)]
    #[from(forward)]
    pub struct String {
        pub concrete: std::string::String,
    }

    impl Convert for super::String {
        type T = String;
        fn convert(self) -> Self::T {
            String::from(self.as_ref())
        }
    }

    impl Convert for String {
        type T = super::String;
        fn convert(self) -> Self::T {
            super::String::from(self.concrete)
        }
    }
}

impl String {
    /// Concatenates `self` with `other`.
    pub fn concat(self, other: Self) -> Self {
        vec![self.as_str(), other.as_str()].join("").into()
    }
    /// Appends `ch` to `self`.
    pub fn append(self, ch: char) -> Self {
        let mut new = self.as_str().to_string();
        new.push(ch);
        new.into()
    }
    /// Returns `true` if `self` contains `other` substring, else `false`.
    pub fn contains(self, other: Self) -> bool {
        self.concrete.contains(other.as_str())
    }
    /// Returns `true` if `self` contains `other` substring, else `false`.
    pub fn truncate(self, new_len: usize) -> String {
        let mut new = self.as_str().to_string();
        new.truncate(new_len);
        new.into()
    }
}
