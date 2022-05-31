type t = {
  next: int;
}

let make () = {
  next = 0;
}

and fresh gen =
  let i = gen.next in
  let gen = { next = i + 1; } in
  (i, gen)
