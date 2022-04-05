# Word Count

Word count is a simple way to count the number of words in a text. The code for calculating a wordcount is as follows.

```arc-lang
{{#include ../../../arc-lang/examples/wordcount.arc:example}}
val word_counts =
  from line in read_stdin(),
       word in line.split(" ")
  yield #{word, count:1}
  keyby word
  window
      duration 10min
      reduce (+) of count
      identity 0;
```
