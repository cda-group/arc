OutputPorts {
  Out1(i32),
  Out2(i32),
}

task Foo(x:i32) (In1(i32)) -> OutputPorts
  on In1(x) =>
    emit In1(2);
    let output =
      if x % 2 == 0 {
        OutputPort::Out1(x)
      } else {
        OutputPort::Out2(x)
      } in
    emit output
end


