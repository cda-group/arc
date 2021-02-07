fun main() {
    ()
}

task Identity() i32 -> i32 {
  fun foo() -> f32 {
    3
  }
    on x => emit foo(x)
}

