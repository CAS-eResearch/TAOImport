#!/usr/bin/env python
import pstats
p = pstats.Stats('taoconvert.prof')
p.sort_stats('cumulative').print_stats(100)
