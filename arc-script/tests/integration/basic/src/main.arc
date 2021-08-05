fun pipe(s: ~{key:i32, value:i32}): ~{key:i32, value:i32} {
    s | Identity()
      | Filter(fun(x): x % 2 == 0)
      | Map(fun(x): x + 1)
}

task Identity(): ~{key:i32, value:i32} -> ~{key:i32, value:i32} {
    on event => emit event;
}

task Map(f: fun(i32): i32): ~{key:i32, value:i32} -> ~{key:i32, value:i32} {
    on event => emit f(event.value) by event.key;
}

task Filter(f: fun(i32): bool): ~{key:i32, value:i32} -> ~{key:i32, value:i32} {
    on event => if f(event.value) {
        emit event
    };
}
