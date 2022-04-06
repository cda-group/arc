# Queries

Arc-Lang allows the formulation of queries over streams (similar to SQL). This concept is borrowed from [Morel](https://github.com/julianhyde/morel) which embeds SQL-style queries over relations in StandardML. The following query defines a word-count application.

```arc-lang
{{#include ../../../arc-lang/examples/wordcount.arc:example}}
```

The top-level statements provided by arc-lang are:
* `from` - Iterate over data items
* `yield` - Map data items
* `where` - Filter data items
* `join` - Join data items on a common predicate
* `group` - Group data items by a key
* `window` - Sliding or tumbling window
* `compute` - Aggregate data items
* `sort` - Sort data items

These statements are described more in detail in the [syntax](../arc-lang/Expr.html#Query).
