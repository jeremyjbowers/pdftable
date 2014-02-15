# pdftable by Kyle Cronan <kyle@pbx.org>
# http://sourceforge.net/projects/pdftable

import sys, glob, operator, re, bisect, csv, htmllib


class Entry:
    """Text entry with coordinates of containing rectangle"""
    
    def __init__(self, top=None, left=None, width=None, height=None,
                 file=None, page=None, content=None):
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.file = file
        self.page = page
        self.page_top = None
        self.page_bottom = None
        self.content = content
        self.non_tabular = False

    def bottom(self):
        return self.top + self.height
    
    def right(self):
        return self.left + self.width

    def from_line(self, line):
        m = re.search(r'top="?(\d+)"?( |>)', line)
        if not m: return None
        self.top = int(m.group(1))
        
        m = re.search(r'left="?(\d+)"?( |>)', line)
        if not m: return None
        self.left = int(m.group(1))
        
        m = re.search(r'width="?(\d+)"?( |>)', line)
        if not m: return None
        self.width = int(m.group(1))
        
        m = re.search(r'height="?(\d+)"?( |>)', line)
        if not m: return None
        self.height = int(m.group(1))
        
        m = re.search(r'>(.*)</text>', line)
        if not m: return None
        self.content = m.group(1)
        
        return self


class EntrySet:
    """Set of Entries from the files we are processing"""
    
    def __init__(self, extractor=None):
        self.extractor = extractor
        self.entries = []
        self.horizontal = HorizontalView(self)
        self.vertical = VerticalView(self)
        self.horizontal.orthogonal = self.vertical
        self.vertical.orthogonal = self.horizontal
        self.offset = 0
        self.max_bottom = 0
        self.current_page = None
        self.current_page_entries = []
        self.current_file = None
        self.html_parser = htmllib.HTMLParser(None)

    def add_entry(self, entry):
        if entry.file != self.current_file or entry.page > self.current_page:
            for previous in self.current_page_entries:
                previous.page_bottom = self.max_bottom
            self.offset = self.max_bottom + 10
            self.current_file = entry.file
            self.current_page = entry.page
            self.current_page_entries = []

        entry.page_top = self.offset
        entry.top += self.offset
        if entry.bottom() > self.max_bottom: self.max_bottom = entry.bottom()
        
        self.entries.append(entry)
        self.current_page_entries.append(entry)
        self.horizontal.index_entries.append((entry.left, entry))
        self.vertical.index_entries.append((entry.top, entry))
    
    def index(self):
        self.horizontal.index_entries.sort(key=lambda r: r[0])
        self.horizontal.index = [r[0] for r in self.horizontal.index_entries]

        self.vertical.index_entries.sort(key=lambda r: r[0])
        self.vertical.index = [r[0] for r in self.vertical.index_entries]
    
    def evaluate(self, first, last):
        """Determine the number of columns contained in this span of rows"""

        self.vertical.begin = first.interval[0]
        self.vertical.end = last.interval[1]
        
        return partition(self.vertical.overlap_curve(),
                         ratio=self.extractor.column_ratio)


class EntryView:
    """Base class for viewing EntrySet rectangles in a single dimension"""
    
    def __init__(self, set, begin=None, end=None):
        self.entryset = set
        self.begin = begin
        self.end = end
        self.index_entries = []
    
    def entries(self):
        if not self.begin: a = 0
        else: a = bisect.bisect_left(self.index, self.begin)
        if not self.end: b = len(self.index)
        else: b = bisect.bisect_right(self.index, self.end-1)
        
        return [r[1] for r in self.index_entries[a:b]]
    
    def parallel_intervals(self, tabular_only=False):
        entries = self.entries()
        if tabular_only: entries = filter(lambda x: not x.non_tabular,
                                          entries)
        return [self.parallel_interval(x) for x in entries]

    def perpendicular_intervals(self):
        return [self.perpendicular_interval(x) for x in self.entries()]

    def interval_entries(self, interval):
        (mini, maxi) = interval
        self.orthogonal.begin, self.orthogonal.end = interval
        return self.orthogonal.entries()

    def extent(self, perpendicular=None):
        if perpendicular: func = self.perpendicular_intervals
        else: func = self.parallel_intervals
        return max([int[0] for int in func()] +
                   [int[1] for int in func()]) + 1
    
    def overlap_curve(self):
        """A measure of total entry overlap as a function of coordinate"""

        ret = []
        intervals = self.parallel_intervals(tabular_only=True)
        if not intervals: return []

        for i in range(self.extent()):
            sum = 0
            count = 0
            for a, b in intervals:
                if a<i and i<b:
                    sum += min(i-a, b-i)
                    count += 1
            # give extra weight for a higher number of overlaps
            if count > 1: sum *= count - 1
            ret.append(sum)
        return ret


class HorizontalView(EntryView):
    """View of the rows in the EntrySet"""
    
    def parallel_interval(self, entry):
        return (entry.top, entry.bottom())
    
    def perpendicular_interval(self, entry):
        return (entry.left, entry.right())


class VerticalView(EntryView):
    """View of the columns in the EntrySet"""

    def parallel_interval(self, entry):
        return (entry.left, entry.right())
    
    def perpendicular_interval(self, entry):
        return (entry.top, entry.bottom())


class Row:
    """Row of entries with links to neighbor rows"""

    def __init__(self, set, interval, entries, prev=None, next=None):
        self.set = set
        self.interval = interval
        self.entries = entries
        self.prev = prev
        self.next = next
        self.done = False

    def non_tabular(self):
        """Single-columns rows beyond a certain width assumed not tabular"""
        
        if len(self.entries) == 1:
            # extent() calculation should covers all rows on this page
            self.set.vertical.begin = self.entries[0].page_top
            self.set.vertical.end = self.entries[0].page_bottom
        
            width = self.entries[0].width
            if (width >= self.set.extractor.max_singlet_width *
                         self.set.vertical.extent()):
                # mark it so that overlap_curve() can skip this entry
                self.entries[0].non_tabular = True
                return True
        return False


class Table:
    """Set of row and column intervals forming a table of entries"""
    
    def __init__(self, extractor, rows, columns):
        self.extractor = extractor
        self.set = extractor.set
        self.rows = rows
        self.columns = columns
    
    def filter(self, str):
        str = re.sub(r'<[^>]+>', '', str)
        self.set.html_parser.save_bgn()
        self.set.html_parser.feed(str)
        return self.set.html_parser.save_end()
    
    def output(self, writer=None, call=None):
        """Output the contents of the table.
        
        Write the lines of CSV data using the given csv.writer class
        (defaults to the extractor's writer) or, if it is specified,
        by calling the 'call' function for each line of data.
        """
        for row in self.rows:
            line_columns = []
            for begin, end in self.columns:
                rowcol = ""
                for entry in row.entries:
                    if entry.left >= begin and entry.left < end:
                        if rowcol: rowcol += " "
                        rowcol += self.filter(entry.content)
                line_columns.append(rowcol)
            if call: call(line_columns)
            else:
                if writer: writer.writerow(line_columns)
                else: self.extractor.writer.writerow(line_columns)


class Extractor:
    """Context for reading PDF data and extracting tables to CSV format.
    
    This module processes the XML data produced by the program 'pdftohtml'.
    Run 'pdftohtml -xml' on your PDF file(s) and feed the result to an
    instance of this class using the read_file or read_line methods.
    
    This module can also be executed as a command line utility.
    """

    def __init__(self, max_singlet_width=0.6, row_ratio=0.3, column_ratio=0.3,
                 max_lines_skip=5, csv_writer=None, output_file=None):
        self.max_singlet_width = max_singlet_width
        self.row_ratio = row_ratio
        self.column_ratio = column_ratio
        self.max_lines_skip = max_lines_skip
        self.set = EntrySet(extractor=self)
        self.page = 0
        self.writer = (csv_writer or
                       csv.writer(output_file or sys.stdout, dialect='excel'))
    
    def read_line(self, line, file=None):
        """Process a line of XML data"""
        
        m = re.search(r'<page .*number="?(\d+)"?( |>)', line)
        if m:
            self.page = int(m.group(1))
        
        m = re.search(r'<text .*</text>', line)
        if m:
            entry = Entry(file=file, page=self.page)
            if entry.from_line(line): self.set.add_entry(entry)
    
    def read_file(self, file):
        """Process all lines from the file object 'file'"""

        for line in file: self.read_line(line, file=file)
    
    def get_rows(self):
        # partition the entry set into rows and form a linked list of entries
        rows = []
        prev = None
        for interval in partition(self.set.horizontal.overlap_curve(),
                                  ratio=self.row_ratio):
            row = Row(self.set, interval,
                      self.set.horizontal.interval_entries(interval))
            rows.append(row)
            row.prev = prev
            if prev: prev.next = row
            prev = row
        return rows
        
    def extract(self):
        """Return a list of tables extracted from the PDF data"""
        
        self.set.index()
        
        rows = self.get_rows()
        if not rows: return []
        
        length = len(rows)
        max_cols = max([len(row.entries) for row in rows])
        tables = []
        # starting with the rows that have the most columns
        for n in range(max_cols, 1, -1):
            complete = False
            # for as long as there are unprocessed rows having n columns
            while not complete:
                complete = True
                # find the first such row
                for row in rows:
                    if row.done or len(row.entries) < n: continue
                    complete = False
                    
                    row.done = True
                    first = row
                    last = row
                    table_columns = None
                    skip_count = 0
                    # and attempt to expand the number of rows in both
                    # possible directions to form a larger table
                    while last.next and not last.next.done:
                        if last.next.non_tabular():
                            skip_count += 1
                            if skip_count > self.max_lines_skip: break
                            last = last.next
                            last.done = True
                            continue
                        intervals = self.set.evaluate(first, last.next)
                        # up to the point where detected # of columns changes
                        if len(intervals) == n: last = last.next
                        else: break
                        table_columns = intervals
                        last.done = True
        
                    skip_count = 0
                    while first.prev and not first.prev.done:
                        if first.prev.non_tabular():
                            skip_count += 1
                            if skip_count > self.max_lines_skip: break
                            first = first.prev
                            first.done = True
                            continue
                        intervals = self.set.evaluate(first.prev, last)
                        if len(intervals) == n: first = first.prev
                        else: break
                        table_columns = intervals
                        first.done = True
                    
                    # record the table if we ended up with at least two rows
                    if first != last:
                        if first.non_tabular():
                            table_rows = []
                        else: table_rows = [first]
                        while first != last:
                            first = first.next
                            if not first.non_tabular():
                                table_rows.append(first)
        
                        tables.append(Table(self, table_rows, table_columns))

        tables.sort(key=lambda t: min([ent.top for ent in t.rows[0].entries]))
        return tables


def partition(curve, ratio=0.3):
    """Return a list of indices that partition the curve into segments.

    Take the local minima in order, greatest values first.
    If the minimum has a value that is more than 'ratio' percent of the value
    of one of the local maxima on either side, then discard that local
    minimum and the smaller of the two adjacent local maxima.
    
    Remaining minima with values no more than 'ratio' percent of both
    adjacent maxima are partition points.
    
    Assumes that curve[0] and curve[-1] are known to be local minima.
    """
    
    last_maximum = None
    minima = []
    
    for i, v in enumerate(curve):
        if not i or i == len(curve)-1: continue
        if v >= curve[i-1] and v > curve[i+1]:
            # local maximum
            if last_maximum:
                (index, value) = last_minimum
                minima.append([index, value, last_maximum, v])
            last_maximum = v
        elif v < curve[i-1] and v <= curve[i+1]:
            # local minimum
            last_minimum = (i, v)
    
    minima.sort(cmp=lambda x,y: cmp(y[1], x[1]))
    
    points=[]
    for i, (index, value, left_max, right_max) in enumerate(minima):
        if value > ratio * left_max or value > ratio * right_max:
            min_diff = None
            min_n = None
            # find the lesser minimum that is closest to this one
            for n in range(i+1, len(minima)):
                n_index = minima[n][0]
                diff = abs(n_index - index)
                if not min_diff or diff < min_diff:
                    min_diff = diff
                    min_n = n
            if min_n:
                # that adjacent minimum inherits this one's max value
                if minima[min_n][0] < index:
                    minima[min_n][3] = right_max
                else:
                    minima[min_n][2] = left_max
            # and then we discard this minimum for not meeting the ratio req.
            continue
        points.append(index)

    points.sort()
    if len(points): ret = [(0, points[0])]
    else: ret = [(0, len(curve)-1)]

    for n in range(len(points)):
        if n == len(points)-1: next = len(curve)-1
        else: next = points[n+1]
        ret.append((points[n], next))
    
    return ret


def command_line():
    import optparse
    opt = optparse.OptionParser(usage="usage: %prog [OPTION]... [FILE]...")
    opt.add_option("-f", "--file", dest="filename", metavar="FILE",
                   help="write output to FILE (may contain %d)")
    opt.add_option("-r", "--row-ratio", dest="row_ratio", type="float",
                   metavar="N", help="eagerness to identify separate" +
                                     "rows, default 0.3 (0 < N < 1)")
    opt.add_option("-c", "--column-ratio", dest="col_ratio", type="float",
                   metavar="N", help="eagerness to identify separate " +
                                     "columns, default 0.3 (0 < N < 1)")
    opt.add_option("-w", "--max-width-ratio", dest="max_width", type="float",
                   metavar="N", help="max percentage width of single-" +
                                     "column rows, default 0.6 (0 < N < 1)")
    opt.add_option("-s", "--max-lines-skip", dest="max_skip", type="int",
                   metavar="N", help="max # of single-column rows before " +
                                     "forcing table separation, default 5")
    (options, args) = opt.parse_args()
    
    for val in (options.row_ratio, options.col_ratio, options.max_width):
        if val is None: continue
        if val < 0 or val > 1: opt.error("ratio must be between 0 and 1")
    if options.max_skip is not None and options.max_skip < 0:
        opt.error("max-lines-skip must be at least 0")
    
    output_file = None
    if options.filename and "%d" not in options.filename:
        output_file = open(options.filename, 'w')

    pdftable = Extractor(row_ratio=options.row_ratio or 0.3,
                         column_ratio=options.col_ratio or 0.3,
                         max_singlet_width=options.max_width or 0.6,
                         max_lines_skip=options.max_skip or 5,
                         output_file=output_file)
    
    if not args: pdftable.read_file(sys.stdin)
    else:
        for file in reduce(operator.add, map(glob.glob, args)):
            f = open(file, 'r')
            pdftable.read_file(f)
            f.close()

    tables = pdftable.extract()
    
    i=1
    for table in tables:
        writer=None
        file = None
        if options.filename and "%d" in options.filename:
            filename = options.filename % i
            i += 1
            file = open(filename, 'w')
            writer = csv.writer(file, dialect='excel')

        table.output(writer=writer)
        if file: file.close

if __name__ == "__main__":
    command_line()
