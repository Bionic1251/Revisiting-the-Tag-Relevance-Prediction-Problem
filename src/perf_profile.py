# Run
# python -m cProfile -o profile.file src/data/make_interim_pickle_files.py
import pstats
from pstats import SortKey
p = pstats.Stats('src/profile.file')
#p.sort_stats(SortKey.CUMULATIVE).strip_dirs().print_stats(40)
p.strip_dirs()
p.sort_stats(SortKey.CUMULATIVE).print_stats(40)
