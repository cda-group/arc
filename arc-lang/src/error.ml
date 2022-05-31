type loc =
  | Loc of (Lexing.position * Lexing.position)
  | NoLoc
  | LocStd of (Lexing.position * Lexing.position)
  | NoLocStd

exception InputError of string
exception LexingError of (loc * string)
exception ParsingError of (loc * string)
exception NamingError of (loc * string)
exception TypingError of (loc * string)

let report loc msg =
  begin
    match loc with
    | Loc (p0, p1) ->
      Printf.eprintf "Error [%s:%d:%d-%d:%d]: %s\n"
        p0.pos_fname
        p0.pos_lnum
        (p0.pos_cnum - p0.pos_bol + 1)
        p1.pos_lnum
        (p1.pos_cnum - p1.pos_bol + 1)
        msg
    | LocStd (p0, p1) ->
      Printf.eprintf "Error (std) [%s:%d:%d-%d:%d]: %s\n"
        p0.pos_fname
        p0.pos_lnum
        (p0.pos_cnum - p0.pos_bol + 1)
        p1.pos_lnum
        (p1.pos_cnum - p1.pos_bol + 1)
        msg
    | NoLoc ->
      Printf.eprintf "Error %s\n" msg
    | NoLocStd ->
      Printf.eprintf "Error (std) %s\n" msg
  end;
  if !Args.verbose then begin
    Printf.printf "%s" (Printexc.get_backtrace ())
  end;
  exit 1
