+++
title = "Arc-Lang"
sort_by = "weight"
+++

**Arc-Lang** is a programming language for processing data streams.

## Features

* Consume and produce streams using different connectors (e.g., Kafka, TCP, files).
* Decode and encode events using different data formats (e.g., CSV, JSON, Avro).
* Interactively create dataflows that transform, filter and aggregate data through concise query syntax.
* Transparently execute programs locally or distributed.
* Inject Rust and Python code into programs.

## Installation

<pre><code>
$ <span class="string">git</span> <span class="string">clone</span> https://github.com/cda-group/arc -b klas/v0.1.0 --single-branch
$ <span class="string">cargo</span> <span class="string">install</span> --path arc/arc-lang
</code></pre>

## Usage

<pre><code>
$ <span class="string">arc-lang</span>
>> print(<span class="string">"Hello world!"</span>);
Hello world!
</code></pre>

## An example

<pre><code>
$ <span class="keyword">for</span> i <span class="keyword">in</span> {<span class="numeric">1</span>..<span class="numeric">100</span>}; <span class="keyword">do</span> <span class="string">echo</span> $RANDOM; <span class="keyword">done</span> > data.csv
$ <span class="string">arc-lang</span>
>> <span class="keyword">from</span> n:<span class="type">i32</span> <span class="keyword">in</span> source(file(<span class="string">"data.csv"</span>), csv())
   <span class="keyword">select</span> {result:n+<span class="numeric">1</span>}
   <span class="keyword">into</span> sink(file(<span class="string">"output.csv"</span>), csv());
</code></pre>

## More

* Checkout the [examples](https://github.com/cda-group/arc/tree/klas/v1/arc-lang/examples) directory for working examples (more to be added soon).
<!-- * Learn more about how to use Arc-Lang in the [Arc-Lang Book](/arc/book/index.html). -->
* View the specification of Arc-Lang in the [Arc-Lang Research Report](/arc/Arc-Lang-Report.pdf).
