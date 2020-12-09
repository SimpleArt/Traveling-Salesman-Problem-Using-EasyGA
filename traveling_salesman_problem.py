import EasyGA
import random
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

set_value = lambda arg: True
close_float = lambda x, y: abs(x-y) < 1e-10

print()
while True:
    try:
        if (dimensions := int(input("Dimensions (note 2 or 3 are plottable) : "))) < 2:
            raise Exception("Invalid, input an integer greater than or equal to 2.")
        if (number_of_points := int(input("Number of points : "))) < 2:
            raise Exception("Invalid, input an integer greater than or equal to 2.")
        cycle_flag = input("Full cycle? (True/False) : ")
        if cycle_flag in ("True", "true"):
            cycle_flag = True
        elif cycle_flag in ("False", "false"):
            cycle_flag = False
        else:
            raise Exception("Invalid, input either True or False.")
        if (max_no_change := int(input("Number of generations without change before stopping : "))) < 2:
            raise Exception("Invalid, input an integer greater than or equal to 1.")
        break
    except Exception as e:
        print(e)
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

# The distance between two points
dist = lambda pnt_1, pnt_2:\
    sum(
        (x1 - x2) ** 2
        for x1, x2
        in zip(pnt_1, pnt_2)
    )

# Create the GA
ga = EasyGA.GA()

# Use permutation methods
ga.permutation_chromosomes()

# Run for at most this many generations for safety
ga.generation_goal = 1000

# Use 5 chromosomes per gene/point
ga.population_size = 10 * number_of_points

# Adapt every 10th generation
ga.adapt_rate = 0.1


def chromosome_impl():
    """Randomly shuffle the data points."""

    new_data = list(data)
    random.shuffle(new_data)
    return new_data

ga.chromosome_impl = chromosome_impl


@EasyGA.Initialization_Methods._chromosomes_to_population
@EasyGA.Initialization_Methods._genes_to_chromosome
@EasyGA.Initialization_Methods._values_to_genes
def initialization_impl(ga):
    """Initialize population by making random chromosomes
    by starting with a random point and then picking
    out the closest point one at a time. (Greedy algorithm)
    Assume there are no duplicate points."""

    new_data = set(data)

    # Select a random starting point.
    last_pnt = new_data.pop()
    yield last_pnt

    # Add on the rest of the points
    for _ in range(len(new_data)):
        best_pnt = None

        # Find the closest point to the last point
        for new_pnt in new_data:
            if (best_pnt is None) or (dist(last_pnt, best_pnt) > dist(last_pnt, new_pnt)):
                best_pnt = new_pnt

        yield best_pnt
        last_pnt = best_pnt
        new_data.discard(last_pnt)

ga.initialization_impl = initialization_impl


# Fitness = distance traveling along the
# points in the chromosome in the given order.
ga.fitness_function_impl = lambda chromosome:\
    sum(
        sqrt(dist(chromosome[i].value, chromosome[i-1].value))
        for i
        in range((0 if cycle_flag else 1), len(chromosome))
    )

ga.target_fitness_type = 'min'


def adapt_population():
    """Adapt the population by applying the greedy algorithm to
    random chromosomes (other than the best few) at random indexes."""

    # Avoid the best chromosome,
    # adapt half of the population.
    for chromosome in ga.population[1::2]:

        # Select a random segment to keep genes from
        index_1 = random.randrange(len(chromosome))
        index_2 = random.randrange(len(chromosome))
        if index_1 > index_2:
            index_1, index_2 = index_2, index_1

        new_data = chromosome[index_1:index_2]

        # Add on the rest of the points
        for _ in range(index_2 - index_1):

            # Randomly add to the left or right side of the segment
            if random.choice([True, False]):
                last_pnt = chromosome[index_1-1]
            else:
                last_pnt = chromosome[index_2]
            
            best_pnt = None

            # Find the closest point to the last point
            for new_pnt in new_data:
                if (best_pnt is None) or (dist(last_pnt.value, best_pnt.value) > dist(last_pnt.value, new_pnt.value)):
                    best_pnt = new_pnt

            # Update segment
            if last_pnt == chromosome[index_1-1]:
                chromosome[index_1] = best_pnt
                index_1 += 1
            else:
                chromosome[index_2-1] = best_pnt
                index_2 -= 1

            new_data.remove(best_pnt)

# Custom adapt population method
ga.adapt_population = adapt_population


best_fitness = 0
best_generation = 0
count = -1

widths = [
    max(
        abs(data[i][k] - data[j][k])
        for i in range(1, len(data))
        for j in range(i)
    )
    for k in range(dimensions)
]

if dimensions == 2:
    fig = plt.figure(figsize = [8, 8])
    ax = fig.add_subplot(111)

elif dimensions == 3:
    fig = plt.figure(figsize = [8, 8])
    ax = fig.add_subplot(111, projection = '3d')

while ga.active() and ga.current_generation < best_generation + max_no_change:
    # Evolve 1 generation
    ga.evolve_generation()

    # Only show if something new happens
    if not close_float(best_fitness, ga.population[0].fitness):
        best_fitness = ga.population[0].fitness
        best_generation = ga.current_generation
        count += 1

        # Show best chromosome
        ga.print_generation()
        print("Best Chromosome \t:")
        for gene in ga.population[0]:
            print(f"\t\t\t  {gene}")
        print(f"Best Fitness \t\t: {ga.population[0].fitness}")
        print()

        # Plot the traveling salesman path
        if dimensions in (2, 3):
            X = [
                [
                    ga.population[0][index].value[i] + (-1)**i * count * widths[i] / (i+1)
                    for index
                    in range(-1 if cycle_flag else 0, len(ga.population[0]))
                ]
                for i
                in range(len(ga.population[0][0].value))
            ]
            ax.plot(*X)
            ax.scatter(*X)

# Show the traveling salesman paths
if dimensions in (2, 3):
    plt.title(f"Progression of the paths found")
    plt.show()
