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

class Cluster:
    """Models cluster information required for scheduling"""
    def __init__(self, num_cpus, compute_slots, data_slots, cache_policy, num_blocks):
        self.num_blocks = num_blocks
        self.cpus = [Cpu(data_slots, self, cache_policy) for i in xrange(0, num_cpus)]
        self.currently_free = set(self.cpus)
        self.compute_slots = compute_slots
        self.cpu_slots_available = {cpu:compute_slots for cpu in self.cpus}
        self.block_placement = [set() for i in xrange(0, num_blocks)]
        self.busy_until = {}
    def insert_in_cpu(self, block, cpu):
        """Track that a block is in the memory for a CPU"""
        self.block_placement[block].add(cpu)
    def evict_from_cpu(self, block, cpu):
        """Block is no longer in local memory"""
        self.block_placement[block].remove(cpu)
    def schedule_task(self, block, current_time, end_time):
        """Find optimal CPU based on what is in memory"""
        # Figure out free resources
        for t in sorted(self.busy_until.keys()):
            if t > current_time:
                break
            self.currently_free = self.currently_free | self.busy_until[t]
            for cpu in self.busy_until[t]:
                self.cpu_slots_available[cpu] += 1
            del self.busy_until[t]
        if end_time not in self.busy_until:
            self.busy_until[end_time] = set()
        cpu = None
        if len(self.currently_free & self.block_placement[block]) > 0:
            cpu = (self.currently_free & self.block_placement[block]).pop()
            assert(self.cpu_slots_available[cpu] > 0)
            self.cpu_slots_available[cpu] -= 1
            if self.cpu_slots_available[cpu] == 0:
                self.currently_free.remove(cpu)
        else:
            try:
                cpu = self.currently_free.pop()
            except:
                print self.cpu_slots_available
                print self.currently_free
                raise
            assert(self.cpu_slots_available[cpu] > 0)
            self.cpu_slots_available[cpu] -= 1
            if self.cpu_slots_available[cpu] > 0:
                self.currently_free.add(cpu)
        self.busy_until[end_time].add(cpu)
        return cpu
        
class Cpu:
    """ Models a CPU with a local memory cache. """
    def __init__(self, data_slots, cluster, cache_policy = 'LRU'):
        # Mapping of block ids stored in memory to the time when the block
        # was last accessed.
        self.cluster = cluster
        self.in_memory_data = {}
        self.data_slots = data_slots
        if cache_policy == 'LRU':
            self.evict = Cpu.lru_evict
            self.update = Cpu.lru_update_block
        elif cache_policy == 'LFU':
            self.evict = Cpu.lfu_evict
            self.update = Cpu.lfu_update_block
    
    def lru_evict(self):
        # Use LRU to determine what to evict.
        oldest_block = -1
        oldest_time = float("inf")
        for data, access_time in self.in_memory_data.items():
            if access_time < oldest_time:
                oldest_block = data
                oldest_time = access_time
        self.cluster.evict_from_cpu(oldest_block, self)
        del self.in_memory_data[oldest_block]
    
    def lru_update_block(self, block, time):
        self.cluster.insert_in_cpu(block, self)
        self.in_memory_data[block] = time

    def lfu_evict(self):
        # LFU to determine what to evict
        least_frequent = sys.maxint
        block_to_evict = -1
        for data, frequency in self.in_memory_data.items():
            if frequency < least_frequent:
                least_frequent = frequency
                block_to_evict = data
        self.cluster.evict_from_cpu(block_to_evict, self)
        del self.in_memory_data[block_to_evict]

    def lfu_update_block(self, block, time):
        if block not in self.in_memory_data:
            self.cluster.insert_in_cpu(block, self)
            self.in_memory_data[block] = 0
        self.in_memory_data[block] += 1

    def maybe_add_data_to_memory(self, block_id, time):
        """ Adds the given block id to the memory associated with this CPU.

        Returns true if the block was added, and false if the block was already
        in memory.
        """
        if block_id in self.in_memory_data:
            self.update(self, block_id, time)
            return False

        if len(self.in_memory_data) >= self.data_slots:
            self.evict(self)

        self.update(self, block_id, time)
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
    if len(argv) < 4:
        print "Usage: simulation.py <numa or cache> <percent_utilization> <file_access_dis_filename>"
        exit(0)

    simulation_function = simulate_numa
    if argv[0] == "cache":
        simulation_function = simulate_cache

    file_access_distribution_filename = argv[3]

    total_time = int(argv[2])


    gnuplot_file = open("results_%s.gp" % argv[0], "w")
    write_gnuplot_template(gnuplot_file)
    
    utilization = float(argv[1])
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
            schedule = []
            cpus_this_time = int(utilization * task_slots_per_cpu * num_cpus) 
            for i in xrange(0, int(total_accesses), cpus_this_time):
               schedule.extend(map(lambda c: (sample(file_access_distribution), i, i + 1), xrange(0, min(int(total_accesses) - len(schedule), cpus_this_time))))
            percent_in_memory = simulation_function(
                    file_access_distribution, num_cpus,
                    task_slots_per_cpu, slots_per_cpu, total_blocks, schedule)
            print percent_in_memory
            results_file.write("%d\t%f\t%d\n" %
                               (slots_per_cpu, percent_in_memory, total_blocks))
        results_file.close()
        if index > 0:
            gnuplot_file.write(",\\\n")
        gnuplot_file.write("'%s' using 1:2 with lp lt %d title '%d CPUs'" %
                           (results_filename, index, num_cpus))

def cdf_to_pdf(dist):
    pdf = {}
    total_so_far = 0.0
    for k in dist:
        pdf[k[0]] = k[1] - total_so_far
        total_so_far =k[1]
    return pdf

def populate_cache_for_numa(cluster, file_access_distribution, num_cpus, data_slots_per_cpu, data_blocks):
    """Prepopulate local memory for NUMA. Using the access distribution, we add likely blocks to memory"""
    pdf = cdf_to_pdf(file_access_distribution)
    total_slots = data_slots_per_cpu * num_cpus
    cpus = cluster.cpus
    slots_left = {cpu: data_slots_per_cpu for cpu in cpus}
    available = num_cpus * data_slots_per_cpu
    for k, v in sorted(pdf.items(), key=lambda pair: pair[1], reverse=True):
        print k, v
        if available == 0:
            break
        space_for_block = int(v * total_slots)
        for ks in sorted(slots_left.keys()):
            assert(slots_left[ks] > 0)
            cluster.insert_in_cpu(k, ks)
            ks.in_memory_data[k] = 0
            slots_left[ks] -= 1
            if slots_left[ks] == 0:
                del slots_left[ks]
            available -= 1
            if available == 0:
                break
            space_for_block -= 1
            if space_for_block == 0:
                break

def simulate_numa(file_access_distribution, num_cpus,
                  task_slots_per_cpu, data_slots_per_cpu, data_blocks, schedule):
    cluster = Cluster(num_cpus, task_slots_per_cpu, data_slots_per_cpu, 'LRU', data_blocks)
    populate_cache_for_numa(cluster, file_access_distribution, num_cpus, data_slots_per_cpu, data_blocks)
    data_blocks_in_memory = 0
    for s in schedule:
        block = s[0]
        start = s[1]
        end = s[2]
        cpu = cluster.schedule_task(block, start, end)
        if block in cpu.in_memory_data:
            data_blocks_in_memory += 1
    percent_in_memory = float(data_blocks_in_memory)/len(schedule)
    return percent_in_memory


def simulate_cache(file_access_distribution,  num_cpus, task_slots_per_cpu,
                   data_slots_per_cpu, data_blocks, schedule):
    cluster = Cluster(num_cpus, task_slots_per_cpu, data_slots_per_cpu, 'LRU', data_blocks)
    iteration = 0
    data_blocks_in_memory = 0
    curr_time = -1
    count = 0
    for s in schedule:
        block = s[0]
        start = s[1]
        end = s[2]
        if start != curr_time:
            curr_time = start
            count = 1
        count += 1
        if not cluster.schedule_task(block, start, end).maybe_add_data_to_memory(block, start):
            data_blocks_in_memory += 1
    percent_in_memory = float(data_blocks_in_memory) / len(schedule)

    return percent_in_memory

    # TODO: Also count the *possible* total amount that count have been in memory
    # (i.e., the # of file accesses on a file that was accessed more than once),
    # just as a baseline.
        
if __name__ == "__main__":
    main(sys.argv[1:])
