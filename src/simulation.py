import random
import sys

def sample(distribution):
    """ Returns a sample from the given distribution, where the distribution
    should be specified as a list of (value, cumulative_probability) tuples,
    in increasing order of cumulative probability. """
    r = random.random()
    for value, cum_prob in distribution:
        if r < cum_prob:
            return value

def get_file_access_distribution(distribution_filename, num_files):
    """ Returns a list of cumulative probabilities that the particular file
    was accesses.
    TODO: that comment makes no sense. """
    access_frequency_counts = []
    distribution_file = open(distribution_filename, "r")
    print "Reading distribution from file %s" % distribution_filename
    for line in distribution_file:
        if line.startswith("File"):
            continue
        items = line.split("\t")
        access_frequency = int(items[0])
        file_cum_prob = float(items[3])
        access_frequency_counts.append((access_frequency, file_cum_prob))
    
    print "Sampling to determine # accesses for each file"
    file_access_counts = []
    total_accesses = 0.0
    while len(file_access_counts) < num_files:
        # Sample from the distribution of file access frequencies to determine
        # how many accesses are for this file.
        num_accesses = sample(access_frequency_counts)
        file_access_counts.append(num_accesses)
        total_accesses += num_accesses

    print "Finding distribution of accesses for each file"
    # Now that we have the number of accesses per file, create a distribution
    # accessses (note that accesses per file is used only as a relative
    # number).
    file_access_distribution = []
    cumulative_accesses = 0.0
    for index, file_access_count in enumerate(file_access_counts):
        cumulative_accesses += file_access_count
        file_access_distribution.append((index,
                                         cumulative_accesses / total_accesses))
    return total_accesses, file_access_distribution

class Cpu:
    """ Models a CPU with a local memory cache. """
    def __init__(self, data_slots):
        # Mapping of block ids stored in memory to the time when the block
        # was last accessed.
        self.in_memory_data = {}
        self.data_slots = data_slots

    def maybe_add_data_to_memory(self, block_id, time):
        """ Adds the given block id to the memory associated with this CPU.

        Returns true if the block was added, and false if the block was already
        in memory.
        """
        if block_id in self.in_memory_data:
            self.in_memory_data[block_id] = time
            return False

        if len(self.in_memory_data) >= self.data_slots:
            # Use LRU to determine what to evict.
            oldest_block = -1
            oldest_time = float("inf")
            for data, access_time in self.in_memory_data.items():
                if access_time < oldest_time:
                    oldest_block = data
                    oldest_time = access_time

            del self.in_memory_data[oldest_block]
        self.in_memory_data[block_id] = time
        return True

def write_gnuplot_template(file):
    file.write("set terminal pdfcairo font 'Gill Sans,20' linewidth 4 "
               "dashed rounded dashlength 2\n")
    file.write("set style line 80 lt 1 lc rgb \"#808080\"\n")
    file.write("set style line 81 lt 0 # dashed\n")
    file.write("set style line 81 lt rgb \"#808080\"  # grey\n")
    file.write("set grid back linestyle 81\n")
    file.write("set border 3 back linestyle 80\n")
    file.write("set xtics nomirror\n")
    file.write("set ytics nomirror\n")
    file.write("set key left\n")

    file.write("set style line 1 lt rgb \"#E41A1C\" lw 2 pt 1\n")
    file.write("set style line 3 lt rgb \"#377EB8\" lw 2 pt 6\n")
    file.write("set style line 2 lt rgb \"#4DAF4A\" lw 2 pt 2\n")
    file.write("set style line 4 lt rgb \"#984EA3\" lw 2 pt 9\n")

    file.write("set output 'results.pdf'\n")
    file.write("set xlabel 'Data slots per CPU' offset 0,0.5\n")
    file.write("set ylabel '% in-memory data accesses' offset 1.5\n")
    file.write("plot ")


def main(argv):
    if len(argv) < 2:
        print "Usage: simulation.py <numa or cache> <file_access_dis_filename>"
        exit(0)

    simulation_function = simulate_numa
    if argv[0] == "cache":
        simulation_function = simulate_cache

    file_access_distribution_filename = argv[1]


    gnuplot_file = open("results_%s.gp" % argv[0], "w")
    write_gnuplot_template(gnuplot_file)

    num_cpus_values = [50, 100, 200, 500, 1000, 2000]
    slots_per_cpu_values = [1, 2, 5, 10]
    total_blocks = 10000
    task_slots_per_cpu = 1
    for index, num_cpus in enumerate(num_cpus_values):
        total_accesses, file_access_distribution = get_file_access_distribution(
                file_access_distribution_filename, total_blocks)

        results_filename = "simulation_results_%s_%d" % (argv[0], num_cpus)
        results_file = open(results_filename, "w")
        for slots_per_cpu in slots_per_cpu_values:
            print ("Running experiment with %d cpus, %d slots per cpu, %d total "
                   "blocks" % (num_cpus, slots_per_cpu, total_blocks))
            cpu_indexes = range(num_cpus)
            # The order in which machine slots before free (assuming 100%
            # utilization, the number of entries here should be
            # TASK_SLOTS_PER_CPU * TOTAL_CPUS). We don't need to store the
            # actual time, for now, since we assume all tasks take the same
            # amount of time; probably will eventually want to make this a
            # priority queue.
            # Right now, TASK_SLOTS_PER_CPU doesn't matter at all, nor does this
            # ordering in cpu_slots_free.
            cpu_slots_free = []
            while len(cpu_slots_free) < task_slots_per_cpu * num_cpus:
                cpu_slots_free.extend(cpu_indexes)
                random.shuffle(cpu_indexes)

            percent_in_memory = simulation_function(
                    total_accesses, file_access_distribution, num_cpus,
                    slots_per_cpu, cpu_slots_free)
            print percent_in_memory
            results_file.write("%d\t%f\t%d\n" %
                               (slots_per_cpu, percent_in_memory, total_blocks))
        results_file.close()
        if index > 0:
            gnuplot_file.write(",\\\n")
        gnuplot_file.write("'%s' using 1:2 with lp lt %d title '%d CPUs'" %
                           (results_filename, index, num_cpus))

def simulate_numa(total_accesses, file_access_distribution, num_cpus,
                  data_slots_per_cpu, cpu_slots_free):
    total_blocks = len(file_access_distribution)
    data_blocks_in_memory = 0
    if simulate_numa:
        # Statically assign data blocks to each CPU.
        data_per_cpu = []
        while len(data_per_cpu) < num_cpus:
            in_memory_data = set()
            while len(in_memory_data) < data_slots_per_cpu:
                in_memory_data.add(random.randint(0, total_blocks - 1))
            data_per_cpu.append(in_memory_data)

        # Run simulation.
        iteration = 0
        while iteration < total_accesses:
            iteration += 1
            cpu_index = cpu_slots_free[iteration % len(cpu_slots_free)]
            data_block = sample(file_access_distribution)
            if data_block in data_per_cpu[cpu_index]:
                data_blocks_in_memory += 1

    percent_in_memory = data_blocks_in_memory * 1.0 / total_accesses
    return percent_in_memory


def simulate_cache(total_accesses, file_access_distribution, num_cpus,
                   data_slots_per_cpu, cpu_slots_free):
    cpus = []
    while len(cpus) < num_cpus:
        cpus.append(Cpu(data_slots_per_cpu))
    # TODO: Should prepopulate cache?

    iteration = 0
    data_blocks_in_memory = 0
    while iteration < total_accesses:
        iteration += 1
        cpu_index = cpu_slots_free[iteration % len(cpu_slots_free)]

        # Assign a data block to the task by sampling from the data block
        # file access distribution.
        data_block = sample(file_access_distribution)
        data_added_to_memory = cpus[cpu_index].maybe_add_data_to_memory(
                data_block, iteration)
        if not data_added_to_memory:
            data_blocks_in_memory += 1

    percent_in_memory = data_blocks_in_memory * 1.0 / total_accesses
    return percent_in_memory

    # TODO: Also count the *possible* total amount that count have been in memory
    # (i.e., the # of file accesses on a file that was accessed more than once),
    # just as a baseline.
        
if __name__ == "__main__":
    main(sys.argv[1:])
