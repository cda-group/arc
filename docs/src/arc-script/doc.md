# Arc-Script

**Arc-Script** is a programming language for data analytics which is based on the concepts of **dataflow programming** and **collection programming**.

## Dataflow programming

**Dataflow programming** is about decomposing complicated problems into independent pipelined **tasks** (e.g., `Map`, `Filter`, `Reduce`). Each task reads data from some input, processes it, and forwards its results to some output. Furthermore, as all tasks work **incrementally** and **concurrently**, dataflow programs can exhibit pipeline parallelism. Together, tasks form a **dataflow graph** which could for example look like this:

```text
--> Filter(fun(x): x % 2) --> Map(fun(x): x + 1) --> Reduce(fun(a, b): a + b) -->
```

## Collection programming

**Collection programming** is about writing programs that are oriented around various types of high-level collections (e.g., Graphs, Streams, Tensors, Relations, DataFrames, DataSets). A **collection** is in a broader sense an abstract data type which allows programmers to model algorithms that involve an arbitrary amount of data points. By being abstract, collections are able to hide complexity about implementation details. Programmers therefore do not need to think about how data inside collections is stored, partitioned, secured, kept consistent, and how operations on it are parallelised.
