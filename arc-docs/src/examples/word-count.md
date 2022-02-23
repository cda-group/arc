# Word Count (Not yet fully supported)


## Implementation

```
val lines = read_stdin();

val word_counts =
  from line in lines,
       word in line.split(" ")
  yield #{word, count:1}
  keyby word
  window
      duration 10min
      reduce (+) of count
      identity 0;
```
