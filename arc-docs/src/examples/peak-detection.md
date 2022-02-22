# [Peak Detection](https://dl.acm.org/doi/pdf/10.1145/3428251) (Not yet fully supported)

> Now, let us consider an algorithm for detect peaks in a stream of numerical 
> values (suppose they are of type `V`). The algorithm searches for the first 
> value that exceeds the threshold `THRESH`. Then, it search for the maximum 
> over the next `#PEAK_CNT` elements, which is considered a peak. After that, 
> the algorithm silences detection for `#SILENCE_CNT` elements to avoid a 
> duplicate detection. This process is repeated indefinitely in order to detect 
> all peaks.

## Implementation

```
def query(stream, PEAK_CNT, THRESH, SILENCE_CNT) {
    stream
        .iterate(fun(stream):
          stream.seq(
          _.search(_ > THRESH),
          _.take(PEAK_CNT)
           .reduce(max)
           .ignore(SILENCE_CNT)
          )
      )
}
```

## Implementation ([StreamQL](https://dl.acm.org/doi/pdf/10.1145/3428251))

```
Q<V,V> start = search(v -> v > THRESH);
Q<V,V> take = take(PEAK_CNT);
Q<V,V> max = reduce((x, y) -> (y > x) ? y : x);
Q<V,V> find1 = pipeline(seq(start, take), max);
Q<V,V> silence = ignore(SILENCE_CNT);
Q<V,V> query = iterate(seq(find1, silence));
```
