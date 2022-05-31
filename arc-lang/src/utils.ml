
exception Panic of string

let todo () = raise (Panic "Not yet implemented")
and unreachable () = raise (Panic "Entered unreachable code")
and panic msg = raise (Panic (Printf.sprintf "%s" msg))

let report_line_error (loc : Lexing.position) msg exit_num =
  print_endline ("File: " ^ Filename.basename loc.pos_fname ^ " Line: " ^ string_of_int loc.pos_lnum);
  print_endline msg;
  exit exit_num

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

let repeat f n ctx =
  let rec repeat f n ctx acc =
    if n = 0 then
      (acc, ctx)
    else
      let (e, ctx) = f ctx in
      repeat f (n-1) ctx (e::acc)
  in
  let (l, ctx) = repeat f n ctx [] in
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
        (l0_init |> List.length)
        (l1_init |> List.length)
    )
  in
  (zip l0_init l1_init acc) |> List.rev

let zip_with f l0_init l1_init =
  zip_with_from f l0_init l1_init []

(* Zip two lists and append them to acc *)
let zip_from l0_init l1_init acc =
  zip_with_from (fun a b -> (a, b)) l0_init l1_init acc

(* Zip two lists and append them to acc *)
let zip l0_init l1_init =
  zip_with_from (fun a b -> (a, b)) l0_init l1_init []

let unzip l =
  let rec unzip l acc0 acc1 =
    match l with
    | [] -> (acc0, acc1)
    | (a, b)::t ->
        unzip t (a::acc0) (b::acc1)
  in
  let (l, r) = unzip l [] [] in
  (l |> List.rev, r |> List.rev)

let get_or x d l = match l |> List.assoc_opt x with
  | Some v -> v
  | None -> d

let duplicates f l =
  let sorted = List.sort (fun a b -> f a b) l in
  let rec duplicates' l acc =
    match l with
    | a::b::t ->
        if a = b then
          duplicates' t (a::acc)
        else
          duplicates' (b::t) acc
    | _ ->
        acc
  in
  duplicates' sorted []

and remove x l =
  let rec remove' l acc =
    match l with
    | h::t ->
        if h = x then
          remove' t acc
        else
          remove' t (h::acc)
    | _ ->
        acc
  in
  let acc = remove' l [] in
  acc |> List.rev

let map = List.map
let map_assoc f = map (function (k, v) -> (k, f v))
let filter = List.filter
let foldl = List.fold_left
let find_map = List.find_map
let assoc = List.assoc
let get x l = match List.assoc_opt x l with
  | Some v -> v
  | None -> raise (Panic (Printf.sprintf "Key %s not found in list [%s]" x (l |> map fst |> String.concat ", ")))
let assoc_opt = List.assoc_opt
let mem = List.mem
let rev = List.rev
let tl = List.tl
let hd = List.hd
let elem = List.mem
let diff l0 l1 = l0 |> filter (fun x -> not (l1 |> elem x))
let dom l = l |> map (fun (a, _) -> a)
let sprintf = Printf.sprintf
let exists = List.exists
and last l = l |> List.rev |> hd
