import sys

def main(argv):
    filename = argv[0]
    f = open(filename, "r")
    frequency_counts = {}
    total_files = 0
    total_accesses = 0
    for line in f:
        frequency = line.count(",")
        if frequency not in frequency_counts:
            frequency_counts[frequency] = 0
        frequency_counts[frequency] += 1
        total_files += 1
        total_accesses += frequency

    output_filename = argv[1]
    output_file = open(output_filename, "w")
    output_file.write("File access frequency\t# Files\tCumulative % Files\t"
                      "Cumulative % Accesses\n")

    print "Writing results to %s" % output_filename
    frequencies = frequency_counts.keys()
    cumulative_total_files = 0.0
    cumulative_total_accesses = 0.0
    for frequency in sorted(frequency_counts.keys()):
        count = frequency_counts[frequency]
        cumulative_total_files += count
        cumulative_total_accesses += count * frequency
        output_file.write("%s\t%s\t%s\t%s\n" %
                          (frequency, count,
                           cumulative_total_files / total_files,
                           cumulative_total_accesses / total_accesses))
    output_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
