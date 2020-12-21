import EasyGA
import random
import matplotlib.pyplot as plt
from math import sqrt, ceil
from mpl_toolkits.mplot3d import Axes3D

set_value = lambda arg: True
close_float = lambda x, y: abs(x-y) < 1e-10

print("""Recommended settings:
2 dimensions
100 points
5 chromosomes
True full cycle
0.04 adapt rate (once every 25 generations)
25 generations without change
250 generations run at most.
""")
while True:
    try:
        if (dimensions := int(input("Dimensions (note 2 or 3 are plottable) : "))) < 2:
            raise ValueError("Invalid, input an integer greater than or equal to 2.")

        if (number_of_points := int(input("Number of points : "))) < 3:
            raise ValueError("Invalid, input an integer greater than or equal to 3.")

        if (population_size := int(input("Number of chromosomes : "))) < 5:
            raise ValueError("Invalid, input an integer greater than or equal to 5.")

        cycle_flag = input("Full cycle? (True/False) : ")
        if cycle_flag in ("True", "true"):
            cycle_flag = True
        elif cycle_flag in ("False", "false"):
            cycle_flag = False
        else:
            raise ValueError("Invalid, input either True or False.")

        if not (0 <= (adapt_rate := float(input("Adapt rate : "))) < 1):
            raise ValueError("Invalid, input a float between 0 and 1. Recommended 0.02 to adapt every 50th generation.")

        if (max_no_change := int(input("Number of generations without change before stopping : "))) < 2:
            raise ValueError("Invalid, input an integer greater than or equal to 2.")

        if (max_generation := int(input("Maximum number of generations run : "))) < 2:
            raise ValueError("Invalid, input an integer greater than or equal to 1.")

        break
    except Exception as e:
        print()
        print(e)
        print()
print()

# Collection of random points
data = [
    tuple(
        random.random()
        for __
        in range(dimensions)
    )
    for _
    in range(number_of_points)
]

# The distance between two genes
dist = lambda gene_1, gene_2:\
    sqrt(sum(
        (x1 - x2) ** 2
        for x1, x2
        in zip(data[gene_1.value], data[gene_2.value])
    ))

# Create the GA
ga = EasyGA.GA()

# Use custom crossover and mutation methods instead of default permutation methods
# See below.
# ga.permutation_chromosomes()

# Run for at most this many generations for safety
ga.generation_goal = max_generation

# Population size given by the user
ga.population_size = population_size

# Don't recompute fitnesses
ga.update_fitness = False

# Minimize the distance
ga.target_fitness_type = 'min'

# Adapt rate given by the user
ga.adapt_rate = adapt_rate

# Don't adapt probability rates
ga.adapt_probability_rate = 0

# Number of parents selected: 5%
ga.parent_ratio = 0.05

# Mutation rates: 25%
ga.gene_mutation_rate = 0.25
ga.chromosome_mutation_rate = 0.25

# Use stochastic arithmetic selection
ga.parent_selection_impl = EasyGA.Parent_Selection.Rank.stochastic_arithmetic

# Mutate best chromosomes and replace the worst with them
ga.mutation_population_impl = EasyGA.Mutation_Methods.Population.best_replace_worst


def greedy_insertion(chromosome, insertion_data):
    """Applies the greedy algorithm by randomly
    inserting points into the given chromosome
    where it minimizes the gained distance.

    Note: The algorithm below involves a lot of
    insertions into the chromosome. Performance
    may be improved by using a more suitable
    data structure for the chromosome instead of
    an array, such as a linked-list."""

    # Insert genes in a random order
    random.shuffle(insertion_data)

    # Add in one gene at a time
    for gene in insertion_data:

        chromosome.add_gene(
            gene,

            # Insert anywhere if there are no genes to compare to
            0 if len(chromosome) == 0 else

            # Find the index of where to insert the gene with minimum distance
            min(

                # Keep track of index with the distance
                enumerate(

                    # Add the gene between chromosome[i-1] and chromosome[i] along with
                    # additional edge cases if we are at the endpoints of the path.
                    sum((
                        dist(gene, chromosome[i])   if i<len(chromosome) else 0,
                        dist(gene, chromosome[i-1]) if i>0 or cycle_flag else 0,
                        -dist(chromosome[i], chromosome[i-1]) if 0<i<len(chromosome) or cycle_flag else 0
                    ))
                    for i
                    in range(len(chromosome)+1-int(cycle_flag))
                ),

                # Minimize by distance
                key = lambda elem: elem[1]

            # Return index
            )[0]
        )

    # Set the new fitness
    chromosome.fitness = ga.fitness_function_impl(chromosome)

    return chromosome


# Fitness = distance traveling along the
# points in the chromosome in the given order.
ga.fitness_function_impl = lambda chromosome:\
    sum(
        dist(chromosome[i], chromosome[i-1])
        for i
        in range(1-int(cycle_flag), len(chromosome))
    )


def crossover(ga, parent_1, parent_2, *args):
    """Cross two parents by keeping parts of the paths
    that they share and then greedily inserting the
    remaining points back in."""

    child = ga.make_chromosome([])
    j = 0

    # Path goes full cycle, so endpoints can be checked
    if cycle_flag:
        removed_genes = []

    # Path does not go full cycle, so endpoints cannot be checked.
    else:
        removed_genes = [parent_1[0], parent_1[-1]]

    # Check each gene from parent 1 in reversed order
    # so that removing unmatched genes from the child
    # does not cause re-indexing issues and becomes
    # more efficient by potentially reducing the
    # number of elements in the list being shifted by
    # each gene removal.
    for i in range([2, 0][int(cycle_flag)], len(parent_1)):

        # Get 3 genes in a row
        genes_1 = [parent_1[i-2], parent_1[i-1], parent_1[i]]

        # Search for the middle gene from parent 1
        # and find the genes next to it in parent 2
        j = parent_2.index_of(genes_1[1], j)
        genes_2 = [parent_2[j-2], parent_2[j-1], parent_2[j]]

        # Check if the middle genes are surrounded by the same genes
        cond_1 = (genes_1[0] == genes_2[0] and genes_1[2] == genes_2[2])
        cond_2 = (genes_1[0] == genes_2[2] and genes_1[2] == genes_2[0])

        # If genes match, add it to the child
        if cond_1 or cond_2:
            child.add_gene(genes_1[1])

        # If genes don't match, remove them
        else:
            removed_genes.append(genes_1[1])

    # Insert all of the removed points.
    return greedy_insertion(
        child,
        removed_genes
    )

ga.crossover_individual_impl = crossover


def mutate(ga, chromosome):
    """Mutate a chromosome by randomly removing genes and re-inserting them."""

    # Randomly remove genes
    removed_genes = [
        chromosome.remove_gene(index)
        for index
        in sorted(
            random.sample(range(len(chromosome)), ceil(ga.gene_mutation_rate*len(chromosome))),
            reverse = True
        )
    ]

    # Reinsert genes
    greedy_insertion(
        chromosome,
        removed_genes
    )

ga.mutation_individual_impl = mutate


def adapt_population():
    """Adapt the population by applying 2-opt to the best chromosome,
    and then attempt to greedily re-insert every chromosome one by one."""

    # Modify the best chromosome
    chromosome = ga.population[0]

    # Try all combinations to reverse, assuming i < j.
    for i in range(1-int(cycle_flag), len(chromosome)):
        for j in range(i+2, len(chromosome)):

            # Compute distances with and without reversing a segment
            original_dist = dist(chromosome[i], chromosome[i-1]) + dist(chromosome[j], chromosome[j-1])
            reversed_dist = dist(chromosome[i], chromosome[j]) + dist(chromosome[i-1], chromosome[j-1])

            # Apply possible optimizations
            if original_dist > reversed_dist:
                chromosome[i:j] = reversed(chromosome[i:j])

    # Greedily re-insert every gene
    for i, gene in enumerate(chromosome):

        chromosome.remove_gene(i)

        greedy_insertion(
            chromosome,
            [gene]
        )

# Custom adapt population method
ga.adapt_population = adapt_population


# Convert data to genes
gene_data = [
    ga.make_gene(i)
    for i
    in range(number_of_points)
]


# Initialize population
ga.population = ga.make_population(
    greedy_insertion(
        ga.make_chromosome([]),
        list(gene_data)
    )
    for _
    in range(ga.population_size)
)


# For keeping track of changes
best_fitness = 0
best_generation = 0
count = -1


# Setup for plotting:
widths = [
    max(
        0.1 + abs(data[i][k] - data[j][k])
        for i in range(1, number_of_points)
        for j in range(i)
    )
    for k in range(dimensions)
]


while ga.active() and ga.current_generation < best_generation + max_no_change:
    # Check if adapt will be used
    adapt_counter = ga.adapt_rate*ga.current_generation
    adapt_check = int(adapt_counter) < int(adapt_counter + ga.adapt_rate)

    # Evolve 1 generation
    ga.evolve(1)

    # Only show if something new happens
    if not close_float(best_fitness, best_fitness := ga.population[0].fitness):
        best_generation = ga.current_generation
        count += 1

        # Show best chromosome
        ga.print_generation()
        ga.print_best_chromosome()
        print()

        # Plot the traveling salesman path
        if dimensions in (2, 3):

            # Create a new figure
            fig = plt.figure(figsize = [7, 7])

            # Add 2 or 3 dimensional subplot
            if dimensions == 2:
                ax = fig.add_subplot(111)

            elif dimensions == 3:
                ax = fig.add_subplot(111, projection = '3d')

            # Show new best path
            if count != 0:
                X = [
                    [
                        data[ga.population[0][index].value][i]
                        for index
                        in range(-int(cycle_flag), len(ga.population[0]))
                    ]
                    for i
                    in range(dimensions)
                ]
                ax.plot(*X, 'co-', alpha = 0.7, label = "new path")

            if count == 0:
                best_chromosome = ga.population[0]
                plt.title('Initial path')

            elif count%10 == 1 and count != 11:
                plt.title(f'{count}st change')

            elif count%10 == 2 and count != 12:
                plt.title(f'{count}nd change')

            elif count%10 == 3 and count != 13:
                plt.title(f'{count}rd change')

            else:
                plt.title(f'{count}th change')

            # Show previous best path
            X = [
                [
                    data[best_chromosome[index].value][i]
                    for index
                    in range(-int(cycle_flag), len(best_chromosome))
                ]
                for i
                in range(dimensions)
            ]
            if count == 0:
                ax.plot(*X, 'co-', alpha = 0.7)
                ax.plot(*X, 'mo-', alpha = 0.7)
            else:
                ax.plot(*X, 'mo--', alpha = 0.7, label = "old path")
                plt.legend(loc = "upper left")

    # Show progress after adapting
    elif adapt_check:

        # Show current generation
        ga.print_generation()
        print()

    best_chromosome = ga.population[0]

# Print results
ga.print_generation()
ga.print_population()

# Print data
print("Data points:")
for gene in best_chromosome:
    print(data[gene.value])

print(f"""
Settings:
{dimensions} dimensions
{number_of_points} points
{population_size} chromosomes
{cycle_flag} full cycle
{adapt_rate} adapt rate (once every {round(1/adapt_rate, 2) if adapt_rate>0 else float('inf')} generations)
{max_no_change} generations without change
{max_generation} generations run at most
""")

# Show the final path
if dimensions in (2, 3):

    # Create a new figure
    fig = plt.figure(figsize = [7, 7])

    # Add 2 or 3 dimensional subplot
    if dimensions == 2:
        ax = fig.add_subplot(111)

    elif dimensions == 3:
        ax = fig.add_subplot(111, projection = '3d')

    # Show final path
    X = [
        [
            data[best_chromosome[index].value][i]
            for index
            in range(-int(cycle_flag), len(best_chromosome))
        ]
        for i
        in range(dimensions)
    ]
    ax.plot(*X, 'co-', alpha = 0.7)
    ax.plot(*X, 'mo-', alpha = 0.7)

    plt.title('Final path')
    plt.show()

ga.graph.lowest_value_chromosome()
plt.title('Length of best path each generation')
plt.ylabel('Length of path')
plt.show()
