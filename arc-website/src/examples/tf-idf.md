# TF-IDF

The formula that is used to compute tf-idf is defined as follows:

* `tf-idf(t, d) = tf(t, d) * idf(t)`
  * `t` is a term
  * `d` is a document in a document set
  * `idf(t) = log [n/df(t)] + 1` is the inverse document frequency
    * `n` is the total number of documents in the document set
    * `df(t)` is the document frequency of `t`
      * i.e., number of documents containing the term `t`
    * `tf(t, d)` is the term frequency of `t` in `d`
      * i.e., number of occurrences of the term `t` within the document `d`
    * `1` is added so that terms which occur in all documents will not be
      entirely ignored.
