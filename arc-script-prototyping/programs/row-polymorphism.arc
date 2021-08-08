fun foo[T](x: %{y:i32|T}): i32 { x.y }
fun bar(x: %{y:i32, z:i32}): i32 { x.y + x.z }

fun hello() {
    val x = %{y:5, z:5};
    val y = %{y:5, z:5, w:9};
    foo(x);
    foo(y);
    bar(x);
}
