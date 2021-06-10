module {
    // Standard Dialect
    func @my_function(%0: i32) -> i32 {
        %1 = std.constant 1 : i32
        %2 = std.addi %0, %1 : i32
        return %2 : i32
    }

    // GPU Dialect
    func @my_gpu_function(%0: f32) -> f32 {
        %1 = "gpu.all_reduce"(%0) ({}) { op = "add" } : (f32) -> (f32)
        %2 = "gpu.all_reduce"(%0) ({
          ^bb(%lhs : f32, %rhs : f32):
              %sum = addf %lhs, %rhs : f32
              "gpu.yield"(%sum) : (f32) -> ()
        }) : (f32) -> (f32)
        return %2 : f32
    }

    // Complex Dialect
    func @my_complex_function(%0: f32, %1: f32) -> complex<f32> {
        %2 = complex.create %0, %1 : complex<f32>
        return %2 : complex<f32>
    }

    // TOSA (Tensor Operation Set Architecture) Dialect
    func @my_tosa_function(%0: tensor<4x4xf32>, %1: tensor<4x4xf32>) -> tensor<4x4xf32> {
        %2 = "tosa.mul"(%0, %1) { shift = 1 : i32 } : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
        return %2 : tensor<4x4xf32>
    }

    // Math Dialect
    func @my_math_function(%0: f32, %1: f32) -> f32 {
        %2 = math.powf %0, %1 : f32
        return %2 : f32
    }
}

