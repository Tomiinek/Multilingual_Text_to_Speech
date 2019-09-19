import argparse
from bisect import bisect_left

def take_closest(l, x):
    pos = bisect_left(l, x)
    if pos == 0: return 0, l[0]
    if pos == len(l): return pos-1, l[-1]
    before = l[pos - 1]
    after = l[pos]
    if after - x < x - before: return pos, after
    else: return (pos - 1), before

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alignment', default=None, type=str, help="File with alignment (.tsvm)", required=True)
    parser.add_argument('-s', '--silence', default=None, type=str, help="File with info about silence (two columns)", required=True)
    parser.add_argument('-t', '--tolerance', default=200, type=int, help="Maximal forward correction of alignment (ms).")
    parser.add_argument('-v', '--overlap', default=100, type=int, help="Duration of a silent span added to every segment.")
    parser.add_argument('-o', '--output', default=None, type=str, help="Output file", required=True)
    args = parser.parse_args()

    starts = []
    ends = []

    with open(args.silence) as silence_file:
        for line in silence_file:
            tokens = line.split()
            # two columnds star end expected
            if len(tokens) > 2: continue
            starts.append(float(tokens[0]))
            if len(tokens) == 1: 
                ends.append(100000)
                continue
            ends.append(float(tokens[1]))

    starts.sort()
    ends.sort()

    overlap = args.overlap / 1000.0

    with open(args.alignment) as alignment_file, open(args.output, 'w') as output_file:
        for line in alignment_file:
            tokens = line.split()
            # three columns start end id expected
            if len(tokens) != 3: continue

            current_start = float(tokens[0])
            current_end = float(tokens[1])

            idx_end, closest_end     = take_closest(ends, current_start)
            idx_start, closest_start = take_closest(starts, current_start)

            if closest_end > current_start:
                # current start is in a silent interal
                if starts[idx_end] < current_start: new_start = max(closest_end - overlap, starts[idx_end])
                # we now know that idx_end == idx_start
                else:
                    assert closest_start == starts[idx_end], ("Something weird has happend!")
                    if abs(current_start - closest_start) > args.tolerance / 1000.0: new_start = current_start
                    else: new_start = max(closest_end - overlap, closest_start)
            else:
                # the alignment is in a noisy interval
                if closest_start > current_start:
                    # the end is too far
                    if abs(current_start - closest_end) > args.tolerance / 1000.0: 
                        # the start is too far
                        if abs(current_start - closest_start) > args.tolerance / 1000.0: new_start = current_start
                        else: new_start = max(ends[idx_start] - overlap, closest_start)
                    # we will move to the end even if we are in a noisy area
                    else: new_start = max(closest_end - overlap, starts[idx_end])
                # we prefer nearest greater end
                else: new_start = max(ends[idx_start] - overlap, closest_start)
               
            idx_end, closest_end     = take_closest(ends, current_end)
            idx_start, closest_start = take_closest(starts, current_end)

            if closest_start < current_end:
                # current start is in a silent interal
                if ends[idx_start] > current_end: new_end = min(closest_start + overlap, ends[idx_start])
                # we now kddnow that idx_end == idx_start
                else:
                    assert closest_end == ends[idx_start], ("Something weird has happend!")
                    if abs(current_end - closest_end) > args.tolerance / 1000.0: new_end = current_end
                    else: new_end = min(closest_start + overlap, closest_end)
            else:
                # the alignment is in a noisy interval
                if closest_end < current_end:
                    # the end is too far
                    if abs(current_end - closest_start) > args.tolerance / 1000.0: 
                        # the start is too far
                        if abs(current_end - closest_end) > args.tolerance / 1000.0: new_end = current_end
                        else: new_end = min(starts[idx_end] + overlap, closest_end)
                    # we will move to the end even if we are in a noisy area
                    else: new_end = min(closest_start + overlap, ends[idx_start])
                # we prefer nearest greater end
                else: new_end = min(starts[idx_end] + overlap, closest_end)

            print('{0}\t{1}\t{2}'.format(new_start, new_end, tokens[2]), file=output_file)