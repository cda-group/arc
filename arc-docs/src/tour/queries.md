# Queries

Arc-Lang allows the formulation of queries over data collections (similar to SQL). This concept is borrowed from [Morel](https://github.com/julianhyde/morel) which embeds SQL-style queries over relations in StandardML. The following query defines a word-count application.

```arc-lang
{{#include ../../../arc-lang/examples/wordcount.arc:example}}
```

The top-level statements provided by Arc-Lang are:

| Operator  | Description                                |
|-----------|--------------------------------------------|
| `from`    | Iterate over data items                    |
| `yield`   | Map data items                             |
| `where`   | Filter data items                          |
| `join`    | Join data items on a common predicate      |
| `group`   | Group data items by a key                  |
| `window`  | Sliding or tumbling window                 |
| `compute` | Aggregate data items                       |
| `reduce`  | Rolling aggregate data items               |
| `sort`    | Sort data items                            |
| `split`   | Split data items into distinct collections |

The exact syntax of these statements is described [here](../arc-lang/Expr.html#Query).

## Query expressions

A *query* in Arc-Lang is a type of expression that takes data collections as operands and evaluates into a data collection as output. A data collection could either be a finite dataframe or an infinite datastream. Queries begin with the `from` keyword and are followed by a set of statements. The `from` statement can in of itself be used to compute the cross product between collections.

```arc-lang
val a = [1,2];
val b = [3];

val c = from x in a, y in b;

# Equivalent to:
# val c = a.flatmap(fun(x) = b.map(fun(y) = #{x:x,y:y}));

assert(c == [#{x:1,y:3},#{x:2,y:3}]);
```

## Projection

The `yield` clause can be used to project elements

```arc-lang
val a = [[[0,1]],[[2,3]],[[4,5]]];

val b = from x in a, y in x yield y+1;

# Equivalent to:
# val b = a.flatmap(fun(x) = x.map(fun(y) = #{x:x,y:y}))
#          .map(fun(r) = r.y + 1);

assert(b == [1,2,3,4,5,6]);
```

## Selection

The `where` clause can be used for retaining elements which satisfy a predicate.

```arc-lang
val a = [0,1,2,3,4];

val b = from x in a where x % 2 == 0;

# Equivalent to:
# val b = a.filter(fun(x) = x % 2 == 0);

assert(a, [0,2,4]);
```

Any kind of expression can be used as a predicate, as long as it evaluates into a boolean value.

## Ordering

The `order` clause can be used to sort elements according to a comparison function. By default, sorting is ascending.

```arc-lang
val a = [3,1,2,4,0];
val b = [#{k:3,v:3},#{k:1,v:2},#{k:2,v:9},#{k:4,v:1},#{k:0,v:3}];

val c = from x in a order x;
val d = from x in a order x desc;
val e = from x in a order x with fun(x,y) = x <= y;
val f = from x in a order x on x.k;

# Equivalent to:
# val c = x._sort(fun(x) = x);
# val d = x._sort(fun(x) = x, _desc: Order::Descending);
# val e = x._sort(fun(x) = x, _compare: fun(x,y) = x <= y);
# val f = x._sort(fun(x) = x, _on: fun(x) = x.k);

assert(c == [0,1,2,3,4]);
assert(d == [4,3,2,1,0]);
assert(e == [0,1,2,3,4]);
assert(f == [#{k:0,v:3},#{k:1,v:2},#{k:2,v:9},#{k:3,v:3},#{k:4,v:1}];
```

Streams cannot be sorted since sorting requires that the input is finite.

## Aggregating

The `compute` clause can be used to aggregate values.

```arc-lang
val a = [1,2,3];
val b = [#{v:1},#{v:2},#{v:3}];

val c = from x in a compute sum;
val d = from x in b compute sum of x.v;

# Equivalent to:
# val c = a._compute(sum);
# val d = a._compute(sum, _of: fun(x) = x.v);

assert(c == #{sum:6});
assert(d == #{sum:6});
```

Aggregation is only allowed on finite-sized data collections.

### Rolling Aggregates

The `reduce` clause can be used to compute rolling aggregates.

```arc-lang
val a = [1,2,3];

val c = from x in a reduce sum;

# Equivalent to:
# val c = a._reduce(sum);

assert(c == [#{sum:1}, #{sum:3}, #{sum:6}]);
```

Rolling aggregates are allowed on both finite- and infinite-sized data collections.

### Custom Aggregators

It is possible to define custom aggregators that can be used inside `compute` clauses. The `sum`, `count`, and `average` aggregators are monoids:

```arc-lang
val count = Aggregator::Monoid(#{
    lift: fun(x) = #{count: 1},
    identity: fun() = #{count: 0},
    merge: fun(x,y) = #{count: x.count+y.count},
    lower: fun(x) = x,
});

val sum = Aggregator::Monoid(#{
    lift: fun(x) = #{sum: x},
    identity: fun() = #{sum: 0},
    merge: fun(x,y) = #{sum: x.sum+y.sum},
    lower: fun(x) = x,
});

val average = Aggregator::Monoid(#{
    lift: fun(x) = #{sum: x, count: 1},
    identity: fun() = #{sum: 0, count: 0},
    merge: fun(x,y) = #{sum: x.sum+y.sum, count: x.count+y.count},
    lower: fun(x) = #{average: x.sum/x.count},
});
```

Aggregators that operate on the whole input such as `median` are holistic:

```arc-lang
val median = Aggregator::Holistic(
    fun(v) = {
        val n = v.len();
        if n % 2 == 0 {
            (v[n/2] + v[n/2+1])/2
        } else {
            v[n/2+1]
        }
    }
);
```

## Grouping

The `group` clause can be used to group elements by key into partitions. After grouping, it is possible to apply a partition-wise transformation.

```arc-lang
val a = [#{k:1,v:2}, #{k:2,v:2}, #{k:2,v:2}];

val b = from x in a group x.k;
val c = from x in a group x.k compute average of x.v;

# Equivalent to:
# val b = a._group(fun(x) = x.k);
# val d = a._group(fun(x) = x.k, _compute: sum, _of: fun(v) = x.v);

assert(b == [#{k:1,v:[2]},#{k:2,v:[2,2]}]);
assert(d == [#{k:1,sum:2},#{k:2,sum:4}]);
```

## Windowing

The `window` clause can be used to slice data collections into multiple, possibly overlapping, partitions. Like `group`, a transformation can be applied per-partition. The `after` clause starts the window at an offset, and the `every` clause specifies the slide of a sliding window.

```arc-lang
val a = [#{t:2022-01-01T00:00:01,v:1}, #{t:2022-01-01T00:00:04,v:2}, #{t:2022-01-01T00:00:09,v:3}];

val b = from x in a window 5s on x.t;
val e = from x in a window 5s on x.t compute count;
val c = from x in a window 5s on x.t after 00:00:02;
val d = from x in a window 5s on x.t every 3s;

# Equivalent to:
# val b = a._window(_length: 5s);
# val b = a._window(_length: 5s, _on: fun(x): x.t, _compute: count);
# val c = a._window(_length: 5s, _on: fun(x): x.t, _after: 2020-01-01T00:00:02);
# val d = a._window(_length: 5s, _on: fun(x): x.t, _every: 3s);

assert(b == [#{t:2020-01-01T00:00:05,v:[1,2]}, #{t:2020-01-01T00:00:10,v:[3]}]);
assert(e == [#{t:2020-01-01T00:00:05,count:2}, #{t:2020-01-01T00:00:10,count:1}]);
assert(c == [#{t:2020-01-01T00:00:07,v:[1,2]}, #{t:2020-01-01T00:00:12,v:[3]}]);
assert(d == [#{t:2020-01-01T00:00:05,v:[1,2]}, #{t:2020-01-01T00:00:08,v:[2]}, #{t:2020-01-01T00:00:11,v:[3]}]);
```

## Joining

The `join` clause can be used to join multiple data collections together whose records satisfy a common predicate. It is logically equivalent to a Cartesian-product between the two collections followed by a selection.

```arc-lang
val a = [#{id:0,name:"Tim"}, #{id:1,name:"Bob"}];
val b = [#{id:0,age:30}, #{id:1,age:100}];

val c = from x in a join y in b on x.id == y.id;

# Equivalent to
# val c = a._join(b, fun(x,y) = x.id == y.id);

assert(c == [#{a:#{id:0,name:"Tim"},b:#{id:0,age:30}}, #{a:#{id:1,name:"Bob"},b:#{id:1,age:100}}]);
```

### Windowed Joins

Joins require at least one of their input collections to be finite. A `join` operation over two streams must be followed by a `window` operation.

```arc-lang
val a = [#{id:0,name:"Tim",t:2020-01-01T00:00:01}, #{id:1,name:"Bob",t:2020-01-01T00:00:09}];
val b = [#{id:0,age:30,t:2020-01-01T00:00:02}, #{id:1,age:100,t:2020-01-01T00:00:04}];

val c =
    from x in a
    join y in b on x.id == y.id
    window 5s on x.t, y.t;

assert(c == [#{a:#{id:0,name:"Tim",t:2020-01-01T00:00:01}, b:#{id:0,age:30,t:2020-01-01T00:00:02}}]);
```

## Splitting

The `split` operator can be used to split a collection into two collections.

```arc-lang
val a = [Ok(1), Err(2), Ok(3)];

val (b, c) = from a split error;
val (d, e) = from a split with fun(x) = match x {
    Ok(v) => Either::L(v),
    Err(v) => Either::R(v),
};

# Equivalent to:
# val (b, c) = a._split(_with: split_error);
# val (d, e) = a._split(_with: fun(x) = match x {
#     Ok(v) => Either::L(v),
#     Err(v) => Either::R(v),
# });

assert(b == [1, 3] and c == [2]);
assert(d == [1, 3] and e == [2]);
```
