#[cfg(all(test, feature = "backend_arcon"))]
mod prost_test1 {
    use prost::Message;
    use prost::Oneof;

    #[derive(Message, Clone, Eq, PartialEq)]
    struct Foo {
        #[prost(int32)]
        a: i32,
        #[prost(message)]
        b: Option<Baz>,
    }

    #[derive(Message, Clone, Eq, PartialEq)]
    struct Baz {
        #[prost(message)]
        c0: Option<Qux>,
        #[prost(message)]
        c1: Option<Qux>,
    }

    #[derive(Message, Clone, Eq, PartialEq)]
    struct Qux {
        #[prost(oneof = "QuxEnum", tags = "1, 2, 3")]
        inner: Option<QuxEnum>,
    }

    #[derive(Oneof, Clone, Eq, PartialEq)]
    enum QuxEnum {
        #[prost(message, tag = "1")]
        Zot(Box<Zot>),
    }

    #[derive(Message, Clone, Eq, PartialEq)]
    struct Zot {
        #[prost(message)]
        d: Option<Foo>,
    }

    #[test]
    fn test() {
        let foo = Foo {
            a: 0,
            b: Baz {
                c0: Qux {
                    inner: QuxEnum::Zot(
                        Zot {
                            d: Foo { a: 3, b: None }.into(),
                        }
                        .into(),
                    )
                    .into(),
                }
                .into(),
                c1: Qux {
                    inner: QuxEnum::Zot(
                        Zot {
                            d: Foo { a: 3, b: None }.into(),
                        }
                        .into(),
                    )
                    .into(),
                }
                .into(),
            }
            .into(),
        };

        let mut buf = Vec::new();

        let expected = foo.clone();

        foo.encode(&mut buf).unwrap();
        let found = Foo::decode(&buf[..]).unwrap();

        assert_eq!(expected, found);
    }
}

#[cfg(all(test, feature = "backend_arcon"))]
mod prost_test2 {
    use prost::Message;
    use prost::Oneof;

    // NOTE: Tags must start from 1 or else prost panics

    #[derive(Message, Clone, Eq, PartialEq)]
    struct List {
        #[prost(oneof = "ListEnum", tags = "1, 2")]
        this: Option<ListEnum>,
    }

    impl ListEnum {
        fn wrap(self) -> List {
            List { this: Some(self) }
        }
    }

    #[derive(Oneof, Clone, Eq, PartialEq)]
    enum ListEnum {
        #[prost(message, tag = "1")]
        Cons(Box<Cons>),
        #[prost(message, tag = "2")]
        Nil(Nil),
    }

    #[derive(Message, Clone, Eq, PartialEq)]
    struct Cons {
        #[prost(int32)]
        val: i32,
        #[prost(message)]
        tail: Option<List>,
    }

    #[derive(Message, Clone, Eq, PartialEq)]
    struct Nil {}

    #[test]
    fn test() {
        let list: List = ListEnum::Cons(
            Cons {
                val: 0,
                tail: ListEnum::Cons(
                    Cons {
                        val: 0,
                        tail: ListEnum::Nil(Nil {}).into(),
                    }
                    .into(),
                )
                .into(),
            }
            .into(),
        );

        let mut buf = Vec::new();

        let expected = list.clone();

        list.encode(&mut buf).unwrap();
        let found = List::decode(&buf[..]).unwrap();

        assert_eq!(expected, found);
    }
}

#[cfg(all(test, feature = "backend_arcon"))]
mod prost_test3 {
    use prost::Message;

    #[derive(Message, Clone, Eq, PartialEq)]
    struct Zib {
        #[prost(message, required)]
        e: (),
    }

    #[test]
    fn test() {
        let z = Zib { e: () };

        let mut buf = Vec::new();

        let expected = z.clone();

        z.encode(&mut buf).unwrap();
        let found = Zib::decode(&buf[..]).unwrap();

        assert_eq!(expected, found);
    }
}
