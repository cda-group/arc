# Queries

Arc-Lang allows the formulation of queries over streams (similar to SQL). This concept is borrowed from [Morel](https://github.com/julianhyde/morel) which embeds SQL-style queries over relations in StandardML. The following query defines a word-count application.

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

write_stdout(word_counts);
```
