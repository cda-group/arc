//! An example of using logos with chumsky to parse sexprs
//! Run it with the following command:
//! cargo run --example logos

// use ariadne::{Color, Label, Report, ReportKind, Source};
// use ast::*;
// use chumsky::{
//     input::{Stream, ValueInput},
//     prelude::*,
// };
// use im_rc::Vector;
// use lexer::tokens::Token;
// use logos::Logos;
//
// trait Tok<'a>: ValueInput<'a, Token = Token, Span = SimpleSpan> {}
// impl<'a, I: ValueInput<'a, Token = Token, Span = SimpleSpan>> Tok<'a> for I {}
//
// trait Syn<'a, I: Tok<'a>, T>: Parser<'a, I, T, extra::Err<Rich<'a, Token>>> {}
// impl<'a, I: Tok<'a>, T, P: Parser<'a, I, T, extra::Err<Rich<'a, Token>>>> Syn<'a, I, T> for P {}
//
// fn parse_name<'a, I: Tok<'a>>() -> impl Syn<'a, I, Name> {
//     select! { Token::Name(x) => x }.labelled("name")
// }
//
// fn parse_field<'a, I: Tok<'a>>() -> impl Syn<'a, I, (Name, Pattern)> {
//     parse_name()
//         .then_ignore(just(Token::Colon))
//         .then(parse_pattern())
// }
//
// fn parse_pattern<'a, I: Tok<'a>>() -> impl Syn<'a, I, Pattern> {
//     recursive(|pattern| {
//         let r#const = r#const().map(PConst);
//         let name = parse_name().map(PVar);
//         let variant = just(Token::Case)
//             .ignore_then(parse_name())
//             .then(pattern)
//             .map(|(x, p)| PVariant(x, p));
//         let tuple = pattern.delimited_by(just(Token::ParenL), just(Token::ParenR));
//         let record = parse_name().then(pattern.or_not()).separated_by(just(Token::Comma)).collect::<Vec<_>>();
//         record
//
//     })
// }
//
// fn parse_path<'a, I: Tok<'a>>() -> impl Syn<'a, I, Path> {
//     let path = parse_name().separated_by(just(Token::ColonColon)).collect::<Vector<_>>();
//     let abs = just(Token::ColonColon).ignore_then(path).map(PAbs);
//     let rel = path.map(PRel);
//     abs.or(rel)
// }
//
// fn parse_type<'a, I: Tok<'a>>() -> impl Syn<'a, I, Type> {
//     recursive(|ty| {
//         todo!()
//
//     })
// }
//
// fn parse_param<'a, I: Tok<'a>>() -> impl Syn<'a, I, (Pattern, Option<Type>)> {
//     parse_pattern()
//         .then(just(Token::Colon).ignore_then(parse_type())).or_not()
// }
//
// fn parse_item<'a, I: Tok<'a>>() -> impl Syn<'a, I, Item> {
//     let def = just(Token::Def).then(parse_param)
// }
//
// fn lit<'a, I: Tok<'a>>() -> impl Syn<'a, I, Lit> {
//     select! {
//         Token::Bool(x) => LBool(x),
//         Token::Float(x) => LFloat(x, None),
//         Token::Int(x) => LInt(x, None),
//         Token::String(x) => LString(x),
//         Token::Char(x) => LChar(x),
//         Token::FloatSuffix((l, x)) => LFloat(l, Some(x)),
//         Token::IntSuffix((l, x)) => LInt(l, Some(x)),
//     }
// }
//
// fn r#const<'a, I: Tok<'a>>() -> impl Syn<'a, I, Const> {
//     select! {
//         Token::Bool(x) => CBool(x),
//         Token::Char(x) => CChar(x),
//         Token::Float(x) => CFloat(x),
//         Token::Int(x) => CInt(x),
//         Token::String(x) => CString(x),
//     }
// }
//
// fn expr<'a, I: Tok<'a>>() -> impl Syn<'a, I, Expr> {
//     recursive(|expr| {
//         let ident = select! { Token::Name(x) => x }.labelled("identifier");
//
//         todo!()
//     })
// }
//
// const SRC: &str = r"
//     (-
//         (* (+ 4.0 7.3) 7.0)
//         (/ 5.0 3.0)
//     )
// ";

fn main() {
    // let token_iter = Token::lexer(SRC)
    //     .spanned()
    //     .map(|(tok, span)| (tok, span.into()));
    //
    // let token_stream = Stream::from_iter(token_iter).spanned((SRC.len()..SRC.len()).into());

    // match expr().parse(token_stream).into_result() {
    //     Ok(sexpr) => println!("{:#?}", sexpr),
    //     Err(errs) => errs.into_iter().for_each(|e| {
    //         Report::build(ReportKind::Error, (), e.span().start)
    //             .with_code(3)
    //             .with_message(e.to_string())
    //             .with_label(
    //                 Label::new(e.span().into_range())
    //                     .with_message(e.reason().to_string())
    //                     .with_color(Color::Red),
    //             )
    //             .finish()
    //             .eprint(Source::from(SRC))
    //             .unwrap()
    //     }),
    // }
}
