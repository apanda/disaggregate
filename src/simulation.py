import random
import sys

# LRU Eviction
# Variable for number of blocks that you can hold in local memory -- can start with 1

# each machine: list of slots, where the size is the number of slots
# and the entry is the time that the slot will be free

# AS WRITTEN, WILL NEED AN INVARIANT ABOUT THE SIZE OF MEMORY: Needs to be at
# least as large as the number of slots
#
#
# have a list of the blocks that are in memory, also variable for the maximum
# number of block slots in memory
#
#
# Simple thing: `100% utilized, when machine gets slot available,  just grab
# another task
# Start by just having a priority queue of events, where the only type of
# event is a task finishing (tuple of task finishing, machine). assume all
# tasks are the same
# length for now, so can just use a list and stick things onto the end.

TASK_SLOTS_PER_CPU = 1
DATA_SLOTS_PER_CPU = 1
TOTAL_CPUS = 100

# A task is just the data that it operates on

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
    def __init__(self):
        # Mapping of block ids stored in memory to the time when the block
        # was last accessed.
        self.in_memory_data = {}

    def maybe_add_data_to_memory(self, block_id, time):
        """ Adds the given block id to the memory associated with this CPU.

        Returns true if the block was added, and false if the block was already
        in memory.
        """
        if block_id in self.in_memory_data:
            self.in_memory_data[block_id] = time
            return False

        if len(self.in_memory_data) >= DATA_SLOTS_PER_CPU:
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

def main(argv): 
    # If false, simulates a cache hierarchy.
    simulate_numa = True
    if argv[0] == "cache":
        simulate_numa = False

    file_access_distribution_filename = argv[1]
    total_blocks = int(argv[2])

    total_accesses, file_access_distribution = get_file_access_distribution(
            file_access_distribution_filename, total_blocks)
    print file_access_distribution

    cpu_indexes = range(TOTAL_CPUS)
    # The order in which machine slots before free (assuming 100% utilization,
    # the number of entries here should be TASK_SLOTS_PER_CPU * TOTAL_CPUS).
    # We don't need to store the actual time, for now, since we assume all
    # tasks take the same amount of time; probably will eventually want to
    # make this a priority queue.
    cpu_slots_free = []
    while len(cpu_slots_free) < TASK_SLOTS_PER_CPU * TOTAL_CPUS:
        cpu_slots_free.extend(cpu_indexes)
        random.shuffle(cpu_indexes)


    data_blocks_in_memory = 0
    if simulate_numa:
        # Statically assign data blocks to each CPU.
        data_per_cpu = []
        while len(data_per_cpu) < TOTAL_CPUS:
            in_memory_data = set()
            while len(in_memory_data) < DATA_SLOTS_PER_CPU:
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
    else:
        cpus = []
        while len(cpus) < TOTAL_CPUS:
            cpus.append(Cpu())
        # TODO: Should prepopulate cache?

        iteration = 0
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
    print percent_in_memory

    # TODO: Also count the *possible* total amount that count have been in memory
    # (i.e., the # of file accesses on a file that was accessed more than once),
    # just as a baseline.
        
if __name__ == "__main__":
    main(sys.argv[1:])

