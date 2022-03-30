{
  open Lexing
  open Token

  exception SyntaxError of string

  let suffix_regex = (Str.regexp "[iufb].+")

  let prefix_suffix lexbuf =
    let s = Lexing.lexeme lexbuf in
    match (Str.full_split suffix_regex s) with
    | [Str.Text prefix; Str.Delim suffix] -> (prefix, suffix)
    | _ -> raise (SyntaxError "Entered unreachable code")

  let suffix lexbuf = prefix_suffix lexbuf |> snd

  let next_line lexbuf =
    let pos = lexbuf.lex_curr_p in
    lexbuf.lex_curr_p <-
      { pos with pos_bol = pos.pos_cnum;
                pos_lnum = pos.pos_lnum + 1
      }
}

let int = ['0'-'9'] ['0'-'9']*
let digit = ['0'-'9']
let frac = '.' digit*
let exp = ['e' 'E'] ['-' '+']? digit+
let float = digit+ frac? exp?
let percentage = digit+ frac? '%'
let char = '\'' [^ '\'' ] '\''
let whitespace = [' ' '\t']+
let newline = '\r' | '\n' | "\r\n"
let name = ['a'-'z' 'A'-'Z' '_'] ['a'-'z' 'A'-'Z' '0'-'9' '_']*
let unit = "unit"
let datetime = int '-' int '-' int ('T' int ':' int ':' int)?
let int_suffix = "i8" | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "u128"
let float_suffix = "f32" | "f64"
let bool_suffix = "bool"
let suffix = int_suffix | float_suffix | bool_suffix

rule main =
  parse
  | "("        { ParenL }
  | ")"        { ParenR }
  | "["        { BrackL }
  | "]"        { BrackR }
  | "#{"       { PoundBraceL }
  | "{"        { BraceL }
  | "}"        { BraceR }
  | "<"        { AngleL }
  | ">"        { AngleR }
(*= Operators ==============================================================*)
  | "!"        { Bang }
  | "!="       { Neq }
  | "%"        { Percent }
  | "*"        { Star }
  | "**"       { StarStar }
  | "+"        { Plus }
  | ","        { Comma }
  | "-"        { Minus }
  | "."        { Dot }
  | ".."       { DotDot }
  | "..="      { DotDotEq }
  | "/"        { Slash }
  | ":"        { Colon }
  | "::"       { ColonColon }
  | ";"        { Semi }
  | "<="       { Leq }
  | "="        { Eq }
  | "=="       { EqEq }
  | "=>"       { Imply }
  | ">="       { Geq }
  | "_"        { Underscore }
  | "|"        { Bar }
  | "@"        { AtSign }
(*= Float extensions =======================================================*)
  | "!=" suffix { NeqSuffix (suffix lexbuf) }
  | "%" suffix  { PercentSuffix (suffix lexbuf) }
  | "**" suffix { StarStarSuffix (suffix lexbuf) }
  | "*" suffix  { StarSuffix (suffix lexbuf) }
  | "+" suffix  { PlusSuffix (suffix lexbuf) }
  | "-" suffix  { MinusSuffix (suffix lexbuf) }
  | "/" suffix  { SlashSuffix (suffix lexbuf) }
  | "<" suffix  { LtSuffix (suffix lexbuf) }
  | "<=" suffix { LeqSuffix (suffix lexbuf) }
  | "==" suffix { EqEqSuffix (suffix lexbuf) }
  | ">" suffix  { GtSuffix (suffix lexbuf) }
  | ">=" suffix { GeqSuffix (suffix lexbuf) }
(*= Keywords ================================================================*)
  | "and"      { And }
  | "as"       { As }
  | "break"    { Break }
  | "band"     { Band }
  | "bor"      { Bor }
  | "bxor"     { Bxor }
  | "class"    { Class }
  | "continue" { Continue }
  | "def"      { Def }
  | "desc"     { Desc }
  | "duration" { Duration }
  | "else"     { Else }
  | "enum"     { Enum }
  | "extern"   { Extern }
  | "for"      { For }
  | "from"     { From }
  | "fun"      { Fun }
  | "group"    { Group }
  | "if"       { If }
  | "in"       { In }
  | "instance" { Instance }
  | "join"     { Join }
  | "loop"     { Loop }
  | "match"    { Match }
  | "mod"      { Mod }
  | "not"      { Not }
  | "on"       { On }
  | "or"       { Or }
  | "order"    { Or }
  | "of"       { Of }
  | "return"   { Return }
  | "reduce"   { Reduce }
  | "step"     { Step }
  | "task"     { Task }
  | "type"     { Type }
  | "val"      { Val }
  | "var"      { Var }
  | "where"    { Where }
  | "while"    { While }
  | "window"   { Window }
  | "use"      { Use }
  | "xor"      { Xor }
  | "yield"    { Yield }
  | "true"     { Bool true }
  | "false"    { Bool false }
(*= Identifiers and Literals ================================================*)
  | int "ns"   { Int (int_of_string (Lexing.sub_lexeme lexbuf lexbuf.lex_start_pos (lexbuf.lex_curr_pos-2)) / 1000000000) }
  | int "us"   { Int (int_of_string (Lexing.sub_lexeme lexbuf lexbuf.lex_start_pos (lexbuf.lex_curr_pos-2)) / 1000000) }
  | int "ms"   { Int (int_of_string (Lexing.sub_lexeme lexbuf lexbuf.lex_start_pos (lexbuf.lex_curr_pos-2)) / 1000) }
  | int 's'    { Int (int_of_string (Lexing.sub_lexeme lexbuf lexbuf.lex_start_pos (lexbuf.lex_curr_pos-1))) }
  | int 'm'    { Int (int_of_string (Lexing.sub_lexeme lexbuf lexbuf.lex_start_pos (lexbuf.lex_curr_pos-1)) / 60) }
  | int 'h'    { Int (int_of_string (Lexing.sub_lexeme lexbuf lexbuf.lex_start_pos (lexbuf.lex_curr_pos-1)) / 60 / 60) }
  | int 'd'    { Int (int_of_string (Lexing.sub_lexeme lexbuf lexbuf.lex_start_pos (lexbuf.lex_curr_pos-1)) / 60 / 60 / 24) }
  | int 'w'    { Int (int_of_string (Lexing.sub_lexeme lexbuf lexbuf.lex_start_pos (lexbuf.lex_curr_pos-1)) / 60 / 60 / 24 / 7) }
  | name       { Name (Lexing.lexeme lexbuf) }
  | int        { Int (int_of_string (Lexing.lexeme lexbuf)) }
  | float      { Float (float_of_string (Lexing.lexeme lexbuf)) }
  | percentage { Float (float_of_string (Lexing.sub_lexeme lexbuf lexbuf.lex_start_pos (lexbuf.lex_curr_pos-1)) /. 100.0) }
  | int int_suffix {
    let (p, s) = prefix_suffix lexbuf in
    IntSuffix ((int_of_string p), s) }
  | float float_suffix {
    let (p, s) = prefix_suffix lexbuf in
    FloatSuffix ((float_of_string p), s) }
  | char       { Char (Lexing.lexeme_char lexbuf 1) }
  | '"'        { string (Buffer.create 17) lexbuf }
  | datetime   { String (Lexing.lexeme lexbuf) }
  | '#'        { line_comment lexbuf; main lexbuf }
  | whitespace { main lexbuf }
  | newline    { next_line lexbuf; main lexbuf }
  | _          { raise (SyntaxError ("Unexpected char: '" ^ (Lexing.lexeme lexbuf) ^ "'")) }
  | eof        { Eof }

and line_comment =
  parse
  | newline { () }
  | _ { line_comment lexbuf }

and string buf =
  parse
  | '"'       { String (Buffer.contents buf) }
  | '\\' '/'  { Buffer.add_char buf '/'; string buf lexbuf }
  | '\\' '\\' { Buffer.add_char buf '\\'; string buf lexbuf }
  | '\\' 'b'  { Buffer.add_char buf '\b'; string buf lexbuf }
  | '\\' 'f'  { Buffer.add_char buf '\012'; string buf lexbuf }
  | '\\' 'n'  { Buffer.add_char buf '\n'; string buf lexbuf }
  | '\\' 'r'  { Buffer.add_char buf '\r'; string buf lexbuf }
  | '\\' 't'  { Buffer.add_char buf '\t'; string buf lexbuf }
  | [^ '"' '\\']+
    { Buffer.add_string buf (Lexing.lexeme lexbuf);
      string buf lexbuf
    }
  | _ { raise (SyntaxError ("Illegal string character: " ^ Lexing.lexeme lexbuf)) }
  | eof { raise (SyntaxError ("String is not terminated")) }
