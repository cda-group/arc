
exception Compiler_error of string

let todo () = raise (Compiler_error "Not yet implemented")
and unreachable () = raise (Compiler_error "Entered unreachable code")
and panic msg = raise (Compiler_error (Printf.sprintf "Panic: %s" msg))

let max_by f l =
  let rec max_by f l mk mv =
    match l with
      | v::t ->
          let k = f v in
          if k > mk then
            max_by f t k v
          else
            max_by f t mk mv
      | [] -> mv in
  match l with
    | v::t -> max_by f t (f v) v
    | [] -> assert false

let mapm f ctx l =
  let (l, ctx) = List.fold_left (
    fun (l, ctx) e ->
      let (e, ctx) = f e ctx in
      (e::l, ctx)
  ) ([], ctx) l in
  (l |> List.rev, ctx)

let mapm_filter f ctx l =
  let (l, ctx) = mapm f ctx l in
  (l |> List.filter_map (fun x -> x), ctx)

let miter f ctx l = List.fold_left (fun ctx e -> f e ctx) ctx l

let values l =
  l |> List.map (fun (_, v) -> v)

let zip_with_from f l0_init l1_init acc =
  let rec zip l0 l1 acc =
    match l0, l1 with
    | h0::t0, h1::t1 ->
        zip t0 t1 ((f h0 h1)::acc)
    | [], [] ->
        acc
    | _, _ -> panic (Printf.sprintf "Zip failed, lists of different length %d and %d"
        (l0 |> List.length)
        (l1 |> List.length)
    )
  in
  zip l0_init l1_init acc

let zip_with f l0_init l1_init =
  zip_with_from f l0_init l1_init []

(* Zip two lists and append them to acc *)
let zip_from l0_init l1_init acc =
  zip_with_from (fun a b -> (a, b)) l0_init l1_init acc

(* Zip two lists and append them to acc *)
let zip l0_init l1_init =
  zip_with_from (fun a b -> (a, b)) l0_init l1_init []

let get_or x d l = match l |> List.assoc_opt x with
  | Some v -> v
  | None -> d

let map = List.map
let filter = List.filter
let foldl = List.fold_left
let find_map = List.find_map
let assoc = List.assoc
let assoc_opt = List.assoc_opt
let mem = List.mem
let rev = List.rev
let tl = List.tl
let hd = List.hd
let elem = List.mem
let diff l0 l1 = l0 |> filter (fun x -> not (l1 |> elem x))
let dom l = l |> map (fun (a, _) -> a)
let sprintf = Printf.sprintf

