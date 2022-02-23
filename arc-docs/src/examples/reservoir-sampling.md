# Reservoir Sampling (Work in progress)

[Source](https://github.com/streaming-algorithms/reservoir_sampling):

> Reservoir sampling is a family of randomized algorithms for choosing a simple random sample without replacement of `k` items from a population of unknown size `n` in a single pass over the items. The size of the population `n` is not known to the algorithm and is typically too large to fit all `n` items into memory. The population is revealed to the algorithm over time, and the algorithm cannot look back at previous items. At any point, the current state of the algorithm must permit extraction of a simple random sample without replacement of size `k` over the part of the population seen so far.

> Let's suppose item `i` appears `m` times in the population. Probability of item `i` being `i` the sample is `P(item i in sample) = k * m / n`.
