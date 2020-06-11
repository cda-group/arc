Following are some rough incomplete ideas for `arc-script`s abstract syntax:

```
S ::= F* E                         # Script
F ::= fn ID (self: T, ...): T = E  # Method
E ::= ID                           # Variable
    | E + E | E * E | ...          # Scalar Binary Ops
    | !E | -E | E as T             # Scalar Unary Ops
    | Int | Float | String         # Scalar Constants
    | Key | Timestamp              # Stream constants
    | [E, ...]                     # Array Constant
    | E.E                          # Array Indexing
    | if E then E else E           # Conditional
    | ID = E; E                    # Assignment
    | E.ID(E, ...)                 # Method call
    | E.B(E, ...)                  # Builtin Combinator call
    | P => E                       # Lambda function
P ::= ID | (ID, ...) | [ID, ...]   # Patterns
T ::= int | float | string         # Scalars
    | time | key | winspec | range # Stream types
    | Stream[T] | Array[T; N, ...] # Collections
    | KeyedStream[K,T]
    | Timer | State
```

# Combinators

This section explores combinators from different libraries. A goal of `arc-script` is to provide comprehensions for different data types which can specialize into such combinators.

## Flink DataStreams

To describe Flink's streaming [combinators](https://ci.apache.org/projects/flink/flink-docs-release-1.10/dev/stream/operators/), the following types are at needed:
* `L[T]` is a list type (1 dimensional array).
* `S[T]` is a stream type.
* `S[T+U]` is a connected stream type.
* `K[T]` is a keyed stream type.
* `W[T]` is a windowed stream type.
* `P[T]` is a partitioned (split) stream type.
* `I[T]` is an iterative stream type.
* `A -> B` is a function-type.
* Other types are generic or scalars.

```
B ::=
  process      # S[T] -> (T -> Timer -> State -> L[U]) -> S[U]
               # A flatmap operation which has access to timers and state.
               #
  map          # S[T] -> (T -> U) -> S[U]
               # Basic map.
               #
  flatmap      # S[T] -> (T -> L[U]) -> S[U]
               # Basic flatmap.
               #
  filter       # S[T] -> (T -> bool) -> S[T]
               # Basic filter.
               #
  keyby        # S[T] -> (T -> key) -> K[U]
               # Partitions the stream through a key extraction function.
               #
  window       # K[T] -> winspec -> W[T]
               # Superimposes a window over the stream according to a specification.
               #
  reduce       # K[T] | W[T] -> (T -> T) -> S[T]
               # Aggregates elements of a stream or window with a semigroup.
               #
  fold         # K[T] | W[T] -> T -> (T -> T) -> S[T]
               # Aggregates elements of a stream or window with a monoid.
               #
  apply        # W[T] -> (L[T] -> U) -> S[U]
               # Holistically aggregates the elements of a windowed stream.
               #
  union        # S[T] -> S[T]+ -> S[T]
               # Unions multiple streams of the same type into one.
               #
  join         # S[T] -> S[T] -> (T -> key) -> (T -> key) -> winspec -> W[T]
               # Inner-joins two streams on a key and time window, pairwise.
               # 
  intervalJoin # K[T] -> K[T] -> range -> S[T]
               # Inner-joins two streams on a key and time window, pairwise.
               # 
  coGroup      # S[T] -> S[T] -> (T -> key) -> (T -> key) -> winspec -> W[T]
               # Inner-joins two streams on a key and time window, listwise.
               #
  connect      # S[T] -> S[U] -> C[T+U]
               # Connects two streams into one with a sum-type.
               # 
  coMap        # S[T+U] -> (T -> X) -> (U -> X) -> S[X]
               # Mapping function over connected streams.
               #
  coFlatmap    # S[T+U] -> (T -> L[X]) -> (U -> L[X]) -> S[X]
               # Flatmap function over connected streams.
               #
  split        # S[T] -> (T -> L[(key, T)]) -> P[T]
               # Splits a stream into multiple by some key.
               #
  select       # P[T] -> key+ -> S[T]
               # Selects one or more streams from a split stream.
               # 
  iterate      # S[T] -> (S[T] -> (L[T], L[T])) -> S[T]
               # Feeds back elements passed in the left tuple field.
               # 
  assign       # S[T] -> (T -> time) -> S[T]
               # Extracts a timestamp from each element.
               # 
  project      # S[T+] -> L[int] -> S[T+]
               # Selects a subset of fields from a stream of tuples.
               # 
  partition    # S[T] -> <UDF> -> S[T]
               # Physically partitions a stream according to a UDF.
               # 
  shuffle      # S[T] -> S[T]
               # Physically partitions a stream randomly.
               # 
  rebalance    # S[T] -> S[T]
               # Physically partitions a stream with round-robin.
               # 
  rescale      # S[T] -> S[T]
               # Partitions elements round-robin to a subset of downstream operators.
               # 
  broadcast    # S[T] -> S[T]
               # Broadcasts elements to every partition.
               # 
  restartChain # S[T] -> S[T]
               # Begin a new function-chain, starting at the previous combinator.
               # 
  doNotChain   # S[T] -> S[T]
               # Do not chain the previous combinator.
               # 
  slotShare    # S[T] -> key -> S[T]
               # Assigns a slot to the stream, which can be shared among streams.
               # 
```

## Rust Iterators

The [iterator cheat sheet](https://danielkeep.github.io/itercheat_baked.html) lists the adaptors below, with the following types:
* `I[T]` is an iterator type.
  - Can either be finite or infinite (should be obvious from the adaptor).
* `O[T]` is an option type.
* `R[T,E]` is a result type.
* `V[T]` is a vector type.
* `N` is a numeral.
* `A -> B` is a function-type.
* `(T,..)` is a tuple type.
* `T`, `U`, `S` are generics.

```
-- Generators --
empty          # () -> I[T]
               # Creates an empty iterator.
               #
iterate        # T -> (T -> I[T]) -> I[T]
               # Creates values by repeatedly applying a UDF to the last value.
               #
once           # T -> I[T]
               # Creates an iterator containing an element.
               #
repeat         # T -> I[T]
               # Creates an iterator over an infinite sequence of repeated values.
               #
repeatCall     # (() -> T) -> I[T]
               # Creates an iterator over an infinite sequence of repeated lazy values.
               #
repeatN        # T -> N -> I[T]
               # Creates an iterator over a finite sequence of repeated values.
               #
unfold         # T -> (T -> O[T]) -> I[T]
               # Creates an iterator by repeatedly unfolding a value.
               #
-- Sequences --
batching       # I[T] -> (&mut I[T] -> O[T]) -> I[T]
               # Consumes multiple elements to produce new ones.
               #
cartesian_prod # I[T] -> I[U] -> I[(T,U)]
               # Cartesian product between two iterators.
               #
chain          # I[T] -> I[T] -> I[T]
               # Appends one iterator to the other.
               #
chunks         # I[T] -> N -> I[I[T]]
               # Turns the iterator into an iterator over iterators.
               #
cmp            # I[T] -> I[T] -> Ordering
partial_cmp    # Lexicographically compares the elements of two iterators
               #
coalesce       # I[T] -> (T -> T -> R[T,(T,T)]) -> I[T]
               # Applies a closure which attempts to merge adjacent elements.
               #
collect        # I[T] -> V[T]
               # Collects an iterator into a vector.
               #
cycle          # I[T] -> I[T]
               # Endlessly repeats an iterator.
               #
dedup          # I[T] -> I[T]
               # Removes all consecutive duplicates from an iterator.
               #
dropping       # I[T] -> N -> I[T]
dropping_back  # Eagerly skips N elements from the front.
               #
enumerate      # I[T] -> I[(int, T)]
               # Gives an iteration count to each element.
               #
eq, ne, ge ..  # I[T] -> I[T] -> bool
               # Returns true if the elements of both iterators are equal.
               #
filter         # I[T] -> (T -> bool) -> I[T]
               # Filters the elements through a predicate.
               #
filter_map     # I[T] -> (T -> O[U]) -> I[U]
               # Filters and maps the elements of the iterator.
               #
flat_map       # I[T] -> (T -> I[U]) -> I[U]
               # Maps and flattens the elements of the iterator.
               #
flatten        # I[I[T]] -> I[T]
               # Flattens one level of nesting of a nested iterator.
               #
group_by       # I[T] -> (&T -> K) -> I[T]
               # Groups elements by an extracted key.
               #
interleave     # I[T] -> I[T] -> I[T]
               # Interleaves elements of two iterators.
               #
int.._shortest # I[T] -> I[T] -> I[T]
               # Interleaves elements of two iterators until one runs out.
               #
intersperse    # I[T] -> T -> I[T]
               # Inserts a value between each element of an iterator.
               #
kmerge         # I[I[T]] -> I[T]
kmerge_by      # Flattens and merges iterators in ascending order.
               #
map            # I[T] -> (T -> U) -> I[U]
               # Applies a transformation to each element individually.
               #
map_results    # I[R[T,E]] -> (T -> U) -> I[R[U,E]]
               # Applies a transformation to all ok values of an iterator over results.
               #
merge          # I[T] -> I[T] -> I[T]
merge_by       # Merges elements of two iterators in ascending order.
               #
pad_using      # I[T] -> N -> (N -> T) -> I[T]
               # Pads the iterator by a minimum length with a closure.
               #
partition      # I[T] -> (T -> bool) -> (I[T], I[T])
partition_map  # Partitions the iterator into two iterators according to a predicate.
               #
rev            # I[T] -> I[T]
               # Reverses an iterator.
               #
scan           # I[T] -> &mut S -> (&mut S -> T -> O[U]) -> I[U]
               # A mix of map and fold.
               #
skip           # I[T] -> N -> I[T]
skip_while     # Skips N number of elements from the front.
               #
sorted         # I[T] -> I[T]
sorted_by      # Sorts elements in ascending order.
               #
step           # I[T] -> N -> I[T]
               # Steps over N elements for each iteration.
               #
take           # I[T] -> N -> I[T]
take_while     # Yields an iterator over the first N elements.
take_while_ref #
               #
tee            # I[T] -> (I[T], I[T])
               # Clones an iterator.
               #
tuple_windows  # I[T] -> I[(T, T,...)]
               # Creates a sliding window of tuples over the iterator.
               #
tuples         # I[T] -> I[(T, T,...)]
               # Creates a tumbling window of tuples over the iterator.
               #
unique         # I[T] -> I[T]
unique_by      # Removes all duplicates from an iterator.
               #
unzip          # I[(T,U)] -> (I[T], I[U])
               # Turns an iterator over pairs into a pair of iterators.
               #
while_some     # I[T] -> I[T]
               # Filters O[T] and stops on the first None.
               #
with_position  # I[T] -> I[Position[T]]
               # Wraps each element with a Position (First, Middle, Last).
               #
zip            # I[T] -> I[U] -> I[(T,U)]
zip_eq         # Turns a pair of iterators into an iterator over pairs.
zip_longest    #
               #
-- Free standing sequences --
               #
cons_tuples    # I[((T1,T2,..),T3,..)] -> I[(T1,T2,..,T3,..)]
               # Flattens tuples in an iterator over tuples.
               #
iproduct       # I[T1] -> I[T2] -> ... -> I[(T1, T2, ...)]
               # Creates an iterator over the Cartesian product of iterators.
               #
izip           # I[T1] -> I[T2] -> ... -> I[(T1, T2, ...)]
               # Zips a variadic number of iterators.
               #
-- Values --
all            # I[T] -> (T -> bool) -> bool
               # Returns true if all elements evaluate to true.
               #
all_equal      # I[T] -> bool
               # Returns true if all elements are equal.
               #
any            # I[T] -> (T -> bool) -> bool
               # Returns true if any element evaluates to true.
               #
count          # I[T] -> N
               # Returns the count of all elements.
               #
find           # I[T] -> (T -> bool) -> T
               # Returns the first element which evaluates to true.
               #
find_position  # I[T] -> (T -> bool) -> (N, T)
               # Returns the first element which evaluates to true and its position.
               #
fold           # I[T] -> S -> (S -> T -> S) -> S
fold_options   # Reduces the elements into a single value using a binary operation.
fold_results   #
fold_while     #
               #
fold1          # I[T] -> (T -> T -> T) -> O[T]
               # Fold without a base case.
               #
join           # I[T] -> &str -> String
               # Combines iterator elements into a string with a separator.
               #
last           # I[T] -> T
               # Returns the last element of an iterator.
               #
max            # I[T] -> T
max_by         # Returns the maximum element
max_by_key     #
min            # 
min_by         #
min_by_key     #
               #
minmax         # I[T] -> (T, T)
minmax_by      # Returns the minimum and maximum element of the iterator.
minmax_by_key  #
               #
next_tuple     # I[T] -> O[(T, T, ...)]
               # Returns the next tuple of elements.
               #
nth            # I[T] -> N -> O[T]
               # Returns the Nth element.
               #
position       # I[T] -> (T -> bool) -> O[N]
rposition      # Returns the position of the first element which evaluates to true.
               #
product        # I[T] -> T
               # Multiplies all elements.
               #
set_from       # &mut I[T] -> I[T] -> I[T]
               # Assign elements of the first iterator with the others.
               #
sum            # I[T] -> T
               # Sums all elements.
               #
-- Other --
assert_equal   # I[T] -> I[T]
               # Assert both iterators match exactly.
               #
by_ref         # I[T] -> I[&T]
               # Iterates over references to the elements to not consume the iterator.
               #
cloned         # I[T] -> I[T]
               # Clones an iterator.
               #
combinations   # I[T] -> N -> I[V[T]]
               # Returns an iterator over all combinations of length N.
               #
diff_with      # I[T] -> I[T] -> O[Diff]
               # Diffs the elements of two iterators.
               #
foreach        # I[T] -> unit
               # Iterates over an iterators elements.
               #
format         # I[T] -> &str -> Format
format_with    # Formats the elements with a separator.
               #
fuse           # I[T] -> I[T]
               # Creates an iterator which always returns None after the next Some.
               #
inspect        # I[T] -> (&T -> unit) -> I[T]
               # Calls a closure on each element
               #
multipeek      # I[T] -> I[T]
               # Allows an iterator to be peeked at any number of times.
               #
multizip       # I[T] -> I[U] -> ... -> I[(T,U,...)]
               # Zips up to 8 elements.
               #
peekable       # I[T] -> I[T]
               # Allows an iterator to be peeked at once per element.
               #
peeking_take_w # I[T] -> (I -> bool) -> I[T]
               # Take while, but peekable.
               #
put_back       # I[T] -> I[T]
               # Creates an iterator where one element can be put back.
               #
put_back_n     # I[T] -> I[T]
               # Creates an iterator where multiple elements can be put back.
               #
rciter         # I[T] -> I[T]
               # Creates an iterator which can be iterated over by multiple handles.
               #
tuple_combinat # I[T] -> [(T, T, ...)]
               # Returns an iterator over all tuple combinations.
               #
```

## Spark DStreams

[Spark DStream API](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.streaming.dstream.DStream)

Most, if not all are supported in Flink in some sense.

## GraphLib Graphs

[GraphLib](https://docs.rs/graphlib/0.6.2/graphlib/iterators/index.html) lists some interesting combinators for graphs, using the following types:

* `G[T]` is a graph type.
* `F` is a float type (weight).

```
new            # -> G[T]
with_capacity  # Creates a new graph
reserve        #
shrink_to_fit  #
               #
capacity       # G[T] -> N
               # Returns the capacity of the graph.
               #
add_vertex     # G[T] -> T -> N
               # Adds a vertex and returns its ID.
               #
add_edge       # G[T] -> N1 -> N2 -> R[..]
               # Adds an edge between vertex N1 and N2.
               #
.._check_cycle # G[T] -> N1 -> N2 -> R[..]
               # Adds an edge and fails if it causes a cycle.
               #
.._with_weight # G[T] -> N1 -> N2 -> F -> R[..]
               # Adds an edge with a weight F.
               #
weight         # G[T] -> N1 -> N2 -> O[F]
               # Returns the weight of an edge if it exists.
               #
set_weight     # G[T] -> N1 -> N2 -> F -> R[..]
               # Sets the weight of an edge if it exists.
               #
has_edge       # G[T] -> N1 -> N2 -> bool
               # Checks if an edge exists.
               #
edge_count     # G[T] -> N
               # Returns the total number of edges in the graph.
               #
vertex_count   # G[T] -> N
               # Returns the total number of vertices in the graph.
               #
fetch          # G[T] -> N -> O[&T]
fetch_mut      # Returns a reference to the vertex with ID N, if present.
               #
remove         # G[T] -> N -> unit
               # Removes a vertex from the graph.
               #
remove_edge    # G[T] -> N -> N -> unit
               # Removes an edge from the graph.
               #
retain         # G[T] -> (&T -> bool) -> unit
               # Filters vertices using the specified predicate.
               #
fold           # G[T] -> S -> (T -> S -> S) -> S
               # Folds vertices in depth-first order.
               #
map            # G[T] -> (&T -> U) -> G[U]
               # Maps vertices of the graph.
               #
is_cyclic      # G[T] -> bool
               # Returns true if the graph contains cycles.
               #
roots_count    # G[T] -> N
               # Returns the number of root-vertices of the graph.
               #
neighbours_c.. # G[T] -> N -> N
in_neighbors_. # Returns the total count of neighboring vertices of a vertex.
out_neighbors_ #
               # 
               # TODO
```

# Sources

## Spark Structured Streaming

* [DataStreamReader/DataSetReader](https://spark.apache.org/docs/latest/api/java/org/apache/spark/sql/streaming/DataStreamReader.html).
* [Kafka]()
*

```
File(path, maxFilesPerTrigger, latestFirst, fileNameOnly)
  text
    wholetext                 # Read file as a single row.
    lineSep                   # Set line separator.
  csv
    sep                       # Set column separator.
    encoding                  # Encoding type.
    quote                     # Set quotation character.
    escape                    # Set quotation escape character.
    charToEscapeQuoteEscaping # Set quotation escape-escape character.
    comment                   # Set comment string.
    header                    # Use first line as header.
    inferSchema               # Enable schema inference.
    ignoreLeadingWhitespace   # Ignore leading whitespace.
    ignoreTrailingWhitespace  # Ignore trailing whitespace.
    nullValue                 # Set null-value string.
    emptyValue                # Set empty-value string.
    nanValue                  # Set non-number-value string.
    positiveInf               # Set positive-infinity-value string.
    negativeInf               # Set negative-infinity-value string.
    dateFormat                # Set date-format.
    timestampFormat           # Set timestamp-format.
    timeZone                  # Set timezone for parsing timestamps.
    maxColumns                # Set column upper-bound-limit.
    maxCharsPerColumn         # Set chars-per-column upper-bound-limit.
    mode                      # How to deal with a corrupt record.
      PERMISSIVE              # Take note of it and null other fields.
      DROPMALFORMED           # Drop it.
      FAILFAST                # Throw an exception.
    columnNameOfCorruptRecord # Set name of corrupt fields.
    multiLine                 # Enable multi-line records.
  json
    primitivesAsString        # Parse primitive values as strings.
    prefersDecimal            # Floating point numbers become decimals (not doubles).
    allowComments             # Allow C-style comments.
    allowUnquotedFieldNames   # Allow unquoted field names.
    allowSingleQuotes         # Allow single quotes.
    allowNumericLeadingZeros  # Allow leading zeroes in numbers.
    allowBackslashEscapingAnyCharacter # Allow backslash to escape any character
    allowUnquotedControlChars # Allow ASCII control-characters
    mode                      # (Same as csv)
    columnNameOfCorruptRecord # (Same as csv)
    dateFormat                # (Same as csv)
    timestampFormat           # (Same as csv)
    timeZone                  # (Same as csv)
    multiLine                 # (Same as csv)
    lineSep                   # (Same as csv)
    dropFieldIfAllNull        # Ignore column if all values are null/empty
  orc                         # (No options)
  parquet
    mergeSchema               # Merge schemas from multiple parquet files.
Socket(host, port)
Rate(rowsPerSecond, rampUpTime, numPartitions) # Data generator
Kafka()
```

# Sinks


# Data Formats & Types

Arc programs contain two components:
* Source and sink schema specification.
* Pipeline logic.

Note:
* Arc should maintain type-safety between both components.
* Type inference should be global.
* Type inference should be dependent.
* To fulfill the previous two requirements, recursion must be prohibited.

Example schema specification:

```
source: foo {
  format: protobuf {
    uri: bar.proto,
  },
  feed: kafka {
    endpoint: localhost:88
  }
}

sink: bar {
  format: json {
    "x": <i32>,
    "y": <i32>,
    "active": <bool>,
    "data": [<i32>]
  },
  feed: file {
    path: bar.json
  }
}
```

Notes:
* The formats (how the data looks like) are with their respective types:
  * [`protobuf`](https://developers.google.com/protocol-buffers/docs/proto)
    * Numbers: `double`, `float`, `int32..64`, `uint32..64`, `sint32..64`, `fixed32..64`, `sfixed32..64`
    * Booleans: `bool`
    * Strings: `string`
    * Bytes: `bytes`
    * Messages: `IDENTIFIER` (Protobuf types)
    * Enums: `enum`
    * Maps: `map<KeyType,ValueType>`
    * Nested types & Parent types
    * Repeated: `repeated` (Field can occur multiple times)
    * Any: `Any` (Any kind of type)
    * Oneof: `oneof` (Only one of the fields can occur)
    * Notes:
      * Types can be optional/required.
      * Types can have default values.
      * Fields have a UID.
      * `enum_allow_alias` permits aliased variants in enums.
  * `json`
    * Numbers: `i8..128`, `u8..128`, `f32..64`
    * Strings: `str`
    * Booleans: `bool`
    * Arrays: `[...]`
    * Objects: `{...}`
    * Null: `null`.
  * `csv`
    * Numbers: `i8..128`, `u8..128`, `f32..64`
    * Strings: `str`
    * Booleans: `bool`
  * [`arrow`](https://github.com/apache/arrow/blob/master/format/Schema.fbs)
    * Null
    * Int
    * FloatingPoint
    * Binary
    * Utf8
    * Bool
    * Decimal
    * Date
    * Time
    * Timestamp
    * Interval
    * List
    * Struct_
    * Union
    * FixedSizeBinary
    * FixedSizeList
    * Map
    * Duration
    * LargeBinary
    * LargeUtf8
    * LargeList
* The endpoints (where the data comes from and goes to) are:
  * File
    * URI
  * Kafka

* Feature: Type providers
  * `arc-script sync` command
  * Occurrences:
    * https://spark.apache.org/docs/latest/sql-data-sources-json.html
    * https://docs.microsoft.com/en-us/dotnet/fsharp/tutorials/type-providers/
  * Types should not only declarable, but also downloadable
    * Specify a protobuf, where types are downloaded directly
    * Specify a JSON/CSV dataset, where data is downloaded (maybe single record) and types are parsed/inferred
    * Note:
      * Data might be compressed, encrypted, remote/local
      * Might be worth specifying some information outside of the code (Arc.toml)?
      * Keep remote type information cached in some build directory
