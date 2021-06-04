module {
    func @my_function(%x_0: i32) -> i32 {
        %x_1 = std.constant 1 : i32
        %x_2 = std.addi %x_0, %x_1 : i32
        return %x_2 : i32
    }
}
