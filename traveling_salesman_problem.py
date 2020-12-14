import EasyGA
import random
import matplotlib.pyplot as plt
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D

set_value = lambda arg: True
close_float = lambda x, y: abs(x-y) < 1e-10

print()
while True:
    try:
        if (dimensions := int(input("Dimensions (note 2 or 3 are plottable) : "))) < 2:
            raise ValueError("Invalid, input an integer greater than or equal to 2.")

        if (number_of_points := int(input("Number of points : "))) < 2:
            raise ValueError("Invalid, input an integer greater than or equal to 2.")

        if (population_size := int(input("Number of chromosomes : "))) < 5:
            raise ValueError("Invalid, input an integer greater than or equal to 5.")

        cycle_flag = input("Full cycle? (True/False) : ")
        if cycle_flag in ("True", "true"):
            cycle_flag = True
        elif cycle_flag in ("False", "false"):
            cycle_flag = False
        else:
            raise ValueError("Invalid, input either True or False.")

        if (max_no_change := int(input("Number of generations without change before stopping : "))) < 2:
            raise ValueError("Invalid, input an integer greater than or equal to 1.")

        if not (0 <= (adapt_rate := float(input("Adapt rate : "))) < 1):
            raise ValueError("Invalid, input a float between 0 and 1.Recommended 0.1.")

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
        in zip(gene_1.value, gene_2.value)
    ))

# Create the GA
ga = EasyGA.GA()

# Use permutation methods
ga.permutation_chromosomes()

# Run for at most this many generations for safety
ga.generation_goal = 10000

# Population size given by the user
ga.population_size = population_size

# Minimize the distance
ga.target_fitness_type = 'min'

# Adapt rate given by the user
ga.adapt_rate = adapt_rate

# Gene mutation rates
ga.gene_mutation_rate = 0.2
ga.max_gene_mutation_rate = 0.4
ga.min_gene_mutation_rate = 0

# Chromosome mutation rates
ga.chromosome_mutation_rate = 0.10
ga.max_chromosome_mutation_rate = 0.15
ga.min_chromosome_mutation_rate = 0.05

# Use stochastic arithmetic selection
ga.parent_selection_impl = EasyGA.Parent_Selection.Rank.stochastic_arithmetic


def greedy_insertion(chromosome, insertion_data):
    """Applies the greedy algorithm by randomly
    inserting points into the given chromosome
    where it minimizes the gained distance."""

    # Insert genes in a random order
    random.shuffle(insertion_data)

    # Add one gene if chromosome is empty
    if len(chromosome) == 0 < len(insertion_data):
        chromosome.add_gene(insertion_data.pop())

    for gene in insertion_data:

        # Add each gene to the chromosome
        # at the index which minimizes the
        # amount of distance it adds to the
        # current path.
        chromosome.add_gene(
            gene,
            sorted(
                list(enumerate(
                    dist(gene, chromosome[i])
                    if (i == 0 and not cycle_flag) else
                    dist(gene, chromosome[i-1])
                    if (i == len(chromosome)) else
                    sum((
                        dist(gene, chromosome[i]),
                        dist(gene, chromosome[i-1]),
                        -dist(chromosome[i], chromosome[i-1])
                    ))
                    for i
                    in range(len(chromosome)+1)
                    if not (i == len(chromosome) and cycle_flag)
                )),
                key = lambda elem: elem[1]
            )[0][0]
        )

    return chromosome


@EasyGA.Initialization_Methods._chromosomes_to_population
def initialization_impl(ga):
    """Initialize population by making random chromosomes
    by starting with a random point and then picking
    out the nearest neighbor repeatedly."""

    return greedy_insertion(
        ga.make_chromosome([]),
        [
            ga.make_gene(pnt)
            for pnt
            in data
        ]
    )

ga.initialization_impl = initialization_impl


# Fitness = distance traveling along the
# points in the chromosome in the given order.
ga.fitness_function_impl = lambda chromosome:\
    sum(
        dist(chromosome[i], chromosome[i-1])
        for i
        in range((0 if cycle_flag else 1), len(chromosome))
    )


def adapt_population():
    """Adapt the population by optimizing the 1st, 3rd, and 4th chromosomes
    by removing 25% of their genes and reinserting them greedily."""

    # For the 1st, 3rd, and 5th chromosomes:
    for i in range(0, 5, 2):

        # Create a copy of the best chromosome
        chromosome = ga.make_chromosome(ga.population[i])

        # Randomly remove a random amount of the chromosome
        removed_genes = [
            chromosome.remove_gene(index)
            for index
            in sorted(
                random.sample(range(len(chromosome)), len(chromosome)//4),
                reverse = True
            )
        ]

        # Replace the worst chromosomes
        ga.population[-i//2-1] = greedy_insertion(
            chromosome,
            removed_genes
        )

# Custom adapt population method
ga.adapt_population = adapt_population


best_fitness = 0
best_generation = 0
count = -1

widths = [
    max(
        0.05 + abs(data[i][k] - data[j][k])
        for i in range(1, number_of_points)
        for j in range(i)
    )
    for k in range(dimensions)
]

if dimensions == 2:
    fig = plt.figure(figsize = [6, 6])
    ax = fig.add_subplot(111)

elif dimensions == 3:
    fig = plt.figure(figsize = [6, 6])
    ax = fig.add_subplot(111, projection = '3d')

while ga.active() and ga.current_generation < best_generation + max_no_change:
    # Evolve 1 generation
    ga.evolve(1)

    best_chromosome = ga.population[0]

    # Only show if something new happens
    if not close_float(best_fitness, best_chromosome.fitness):
        best_fitness = best_chromosome.fitness
        best_generation = ga.current_generation
        count += 1

        # Show best chromosome
        ga.print_generation()
        print("Best Chromosome \t:")
        for gene in best_chromosome:
            print(f"\t\t\t  {gene}")
        print(f"Best Fitness \t\t: {best_chromosome.fitness}")
        print()

        # Plot the traveling salesman path
        if dimensions in (2, 3):
            X = [
                [
                    best_chromosome[index].value[i] + (-1)**i * count * widths[i] / (i+1)
                    for index
                    in range(-1 if cycle_flag else 0, len(best_chromosome))
                ]
                for i
                in range(dimensions)
            ]
            ax.plot(*X)
            ax.scatter(*X)

    # Show progress after adapting
    elif int(adapt_counter := ga.adapt_rate*ga.current_generation) > int(adapt_counter - ga.adapt_rate):

        # Show best chromosome
        ga.print_generation()
        print("Best Chromosome \t:")
        for gene in best_chromosome:
            print(f"\t\t\t  {gene}")
        print(f"Best Fitness \t\t: {best_chromosome.fitness}")
        print()

ga.print_generation()
ga.print_population()

# Show the traveling salesman paths
if dimensions in (2, 3):
    plt.title(f"Progression of the paths found")
    plt.show()
