# Streams

A **stream** is a potentially infinite sequence of **events**.

```
Type ::=
  | '~' Type  # A stream of events of a specific Type
```

## Example

This shows an example of an identity function over streams of integers. Streams are a central concept of Arc-Script programs and are therefore given special syntax in the form of `~`.

```
fun identity(s: ~i32): ~i32 {
    s
}
```
