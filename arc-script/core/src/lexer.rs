use logos::Logos;

#[derive(Logos, Debug, PartialEq)]
enum Token {
    #[regex("\n")]
    #[error]
    Error,
}
