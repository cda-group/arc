# XFAIL: *
# RUN: arc-lang %s
from line:String in source(file("data/generated/lines.txt"), csv(','), ingestion())
from word in line.split(" ")
over tumbling(size=1h) {
  group word {
    compute {
      count
    } as c
  } as g
} as w
select {w.g.word, w.g.c.count}
into file("prices.csv"), csv(','));
