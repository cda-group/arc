# Durations

A **duration** is a *length of time*.

```text
Literal ::=
  | [0-9]+'ns'  # Nanosecond
  | [0-9]+'us'  # Microsecond
  | [0-9]+'ms'  # Millisecond
  | [0-9]+'s'   # Second
  | [0-9]+'m'   # Minute
  | [0-9]+'h'   # Hour
  | [0-9]+'d'   # Day
  | [0-9]+'w'   # Week
  | ..

Scalar ::=
  | 'duration'
  | ..
```

## Examples

Some examples:

```text
100ns
100ms
52w
100d
```
