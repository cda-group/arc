from line:String in source(file("lines.txt"), csv(','), ingestion())
from word in line.split(" ")
group word {
  over tumbling(size=1h) {
    compute count of word as count
  } as w
} as g
select {g.word, g.w.c.count}
into file("prices.csv"), csv(','));
