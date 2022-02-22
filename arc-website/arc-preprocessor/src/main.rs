use clap::{App, Arg, ArgMatches, SubCommand};
use mdbook::book::Book;
use mdbook::errors::Error;
use mdbook::preprocess::{CmdPreprocessor, Preprocessor, PreprocessorContext};
use std::io;
use std::process;

pub fn make_app() -> App<'static, 'static> {
    App::new("arc-preprocessor")
        .about("A mdbook preprocessor for arc")
        .subcommand(
            SubCommand::with_name("supports")
                .arg(Arg::with_name("renderer").required(true))
                .about("Check whether a renderer is supported by this preprocessor"),
        )
}

fn main() {
    let matches = make_app().get_matches();

    let preprocessor = ArcLang::new();

    if let Some(sub_args) = matches.subcommand_matches("supports") {
        handle_supports(&preprocessor, sub_args);
    } else if let Err(e) = handle_preprocessing(&preprocessor) {
        eprintln!("{}", e);
        process::exit(1);
    }
}

fn handle_preprocessing(pre: &dyn Preprocessor) -> Result<(), Error> {
    let (ctx, book) = CmdPreprocessor::parse_input(io::stdin())?;

    if ctx.mdbook_version != mdbook::MDBOOK_VERSION {
        eprintln!(
            "Warning: The {} plugin was built against version {} of mdbook, \
             but we're being called from version {}",
            pre.name(),
            mdbook::MDBOOK_VERSION,
            ctx.mdbook_version
        );
    }

    let processed_book = pre.run(&ctx, book)?;
    serde_json::to_writer(io::stdout(), &processed_book)?;

    Ok(())
}

fn handle_supports(pre: &dyn Preprocessor, sub_args: &ArgMatches) -> ! {
    let renderer = sub_args.value_of("renderer").expect("Required argument");
    let supported = pre.supports_renderer(&renderer);

    if supported {
        process::exit(0);
    } else {
        process::exit(1);
    }
}

pub struct ArcLang;

impl ArcLang {
    pub fn new() -> ArcLang {
        ArcLang
    }
}

const KEYWORDS: &[&str] = &[
    "and", "or", "xor", "band", "bor", "bxor", "is", "not", "in", "class", "instance", "def",
    "task", "on", "emit", "val", "var", "fun", "fun", "unit", "mod",
];

impl Preprocessor for ArcLang {
    fn name(&self) -> &str {
        "arc-preprocessor"
    }

    fn run(&self, _: &PreprocessorContext, mut book: Book) -> Result<Book, Error> {
        preprocess_grammar(&mut book);
        preprocess_snippet(&mut book);

        Ok(book)
    }

    fn supports_renderer(&self, renderer: &str) -> bool {
        renderer != "not-supported"
    }
}

fn preprocess_grammar(book: &mut Book) {
    let grammar_regex = regex::Regex::new(r"(?s)```grammar\n(.*?)```").unwrap();
    let head_regex = regex::Regex::new(r"([A-Z][A-Za-z]*)( ::=.*)").unwrap();
    let keyword_regex = regex::Regex::new(r#""([^ ]+?)""#).unwrap();
    let keyword_subst = r"<b>${1}</b>";
    let nonterm_regex = regex::Regex::new(r"\[([A-Z][A-Za-z]+)\]").unwrap();
    let nonterm_subst = r#"<a href="${1}.html#${1}">${1}</a>"#;
    let subterm_regex = regex::Regex::new(r"\[\[([A-Z][A-Za-z]+)\]\]").unwrap();
    let comment_regex = regex::Regex::new(r"( *(?:\||::=).*)#[^{](.*)").unwrap();
    let comment_subst = r#"$1<i style="color:gray">${2}</i>"#;
    book.for_each_mut(|item| {
        if let mdbook::BookItem::Chapter(ch) = item {
            let path = ch.path.as_ref().unwrap();
            let name = path.file_stem().unwrap().to_str().unwrap();
            ch.content = grammar_regex
                .replace_all(&ch.content, |caps: &regex::Captures<'_>| {
                    let subterm_subst = format!(r#"<a href="{}.html#${{1}}">${{1}}</a>"#, name);
                    let head_subst =
                        format!(r#"<a id="${{1}}" href="{}.html#${{1}}">${{1}}</a>$2"#, name);
                    let s = caps.get(1).unwrap().as_str();
                    let s = keyword_regex.replace_all(&s, keyword_subst);
                    let s = comment_regex.replace_all(&s, comment_subst);
                    let s = subterm_regex.replace_all(&s, subterm_subst);
                    let s = nonterm_regex.replace_all(&s, nonterm_subst);
                    let s = head_regex.replace_all(&s, head_subst);
                    format!("<pre><code>{}</code></pre>", s)
                })
                .into_owned();
        }
    });
}

fn preprocess_snippet(book: &mut Book) {
    let grammar_regex = regex::Regex::new(r"(?s)```arc-lang(-todo)?\n(.*?)```").unwrap();
    let comment_regex = regex::Regex::new(r"#[^{].*").unwrap();
    let comment_subst = r#"<i style="color:gray">${0}</i>"#;
    let keyword_regex = regex::Regex::new(&format!(
        r"(^|\n|[^[:alnum:]_])({})($|\n|[^[:alnum:]_])",
        KEYWORDS.join("|"),
    ))
    .unwrap();
    let keyword_subst = r"${1}<b>${2}</b>${3}";
    book.for_each_mut(|item| {
        if let mdbook::BookItem::Chapter(ch) = item {
            ch.content = grammar_regex
                .replace_all(&ch.content, |caps: &regex::Captures<'_>| {
                    let s = caps.get(2).unwrap().as_str();
                    let s = keyword_regex.replace_all(&s, keyword_subst);
                    let s = comment_regex.replace_all(&s, comment_subst);
                    if caps.get(1).is_some() {
                        format!(
                            r#"<pre><code style="background-color:#FFC590">{}</code></pre>"#,
                            s
                        )
                    } else {
                        format!("<pre><code>{}</code></pre>", s)
                    }
                })
                .into_owned();
        }
    });
}
