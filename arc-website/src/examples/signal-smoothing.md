# [Signal Smoothing](https://dl.acm.org/doi/pdf/10.1145/3428251)

> Assume that the input stream consists of signal measurements of type `V` (integer type) which are collected at a fixed frequency. We will consider a computation that is the composition of a smoothing filter and calculating the derivative. We use a low-pass filter to smooth the input into results `f : F` (floating point type), where `f = (v1 + 2*v2 + 4*v3 + 2*v4 + v5)/10` for each five consecutive input items `v1, v2, ..., v5`. Then, we compute the derivative `d : D` (floating point type) where `d = f2 − f1` for every two consecutive smoothed values.

## Implementation

```
fun query(stream) {
    stream
      .swindow(5, 1, fun(v): (v[0] + 2*v[1] + 4*v[2] + 2*v[3] + v[4]) / 10.0)
      .swindow(2, 1, fun(f): f[1] - f[0])
}
```

And

```
from s in stream
window
  length 5
  stride 1
  compute fun(v): (v[0] + 2*v[1] + 4*v[2] + 2*v[3] + v[4]) / 10.0
window
  length 2
  stride 1
  compute fun(f): f[1] - f[0]
```

## Implementation ([StreamQL](https://dl.acm.org/doi/pdf/10.1145/3428251))

```
Q<V,F> smooth = sWindow(5, 1, (a, b, c, d, e) -> (a + 2*y + 4*c + 2*d + e) / 10.0);
Q<F,D> deriv = sWindow(2, 1, (a, b) -> b - a);
Q<V,D> query = pipeline(smooth, deriv);
```
