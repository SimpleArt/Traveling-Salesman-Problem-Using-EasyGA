from EasyGA import *
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import matplotlib.pyplot as plt


class TSP_Solver(GA):
    """Solver for the (Euclidean) traveling salesman problem using genetic algorithms.

    Create the solver:
        solver = TSP_Solver(data, {cycle, *, population_size, adapt_rate, max_generation, max_no_change})

    Run the solver:
        solver.evolve()

    Get the result as a list of indexes of the data:
        path = list(solver.population[0])

    Non-Euclidean distances may be used by defining a custom solver.dist method.

    See https://github.com/danielwilczak101/EasyGA/wiki for more details on how to use EasyGA.
    """

    #=============================#
    # TSP_Solver Built-in Methods #
    #=============================#

    parent_selection_impl = Parent.Rank.stochastic_arithmetic
    mutation_population_impl = Mutation.Population.best_replace_worst

    #================#
    # Initialization #
    #================#

    def __init__(
            self,
            data,
            cycle = True,
            *,  # since the following attributes are optional and only related to the GA
                # instead of the TSP, they must be given as a kwarg instead of an arg.
            population_size = 10,
            adapt_rate = 0.05,
            max_generation = 300,
            max_no_change = 50
        ):
        """Custom initialization method."""

        # Setup with default init
        super().__init__()

        # Set attributes
        self.population_size = population_size
        self.adapt_rate = adapt_rate
        self.cycle = int(cycle)
        self.update_fitness = False
        self.target_fitness_type = 'min'
        self.adapt_probability_rate = 0
        self.parent_ratio = 0.05
        self.best_generation = 0
        self.best_chromosome = None

        # Termination attributes
        self.generation_goal = max_generation
        self.change_goal     = max_no_change

        # Create randomized data
        self.data = data


    def initialize_population(self):
        """Initialize the population using greedy insertion."""

        # Convert data to genes
        gene_data = self.make_chromosome(range(number_of_points))

        # Initialize the population
        self.population = self.make_population(
            self.greedy_insertion(
                self.make_chromosome([]),
                list(gene_data)
            )
            for _
            in range(self.population_size)
        )

    #========================#
    # Fitness related methods#
    #========================#

    def dist(self, gene_1, gene_2):
        """Euclidean norm between two points given by sqrt((x1-x2)^2 + (y1-y2)^2 + ...).

        Note if overriding this method: to get the data points being referred to, use
            data_point = data[gene.value]
        """

        return np.sqrt(sum(
            (x1 - x2) ** 2
            for x1, x2
            in zip(data[gene_1.value], data[gene_2.value])
        ))


    def fitness_function_impl(self, chromosome):
        """Distance along the path."""

        return sum(
            self.dist(chromosome[i], chromosome[i-1])
            for i
            in range(1-self.cycle, len(chromosome))
        )

    #============#
    # Heuristics #
    #============#

    def greedy_insertion(self, chromosome, insertion_data):
        """Applies a greedy algorithm by randomly inserting points into the given
        chromosome where it minimizes the added distance."""

        # Insert genes in a random order
        random.shuffle(insertion_data)

        # Add in one gene at a time
        for gene in insertion_data:

            chromosome.insert(

                # Insert at 0 if there are no genes
                0 if len(chromosome) == 0 else

                # Find the index of where to insert the gene with minimum distance
                min(

                    # Keep track of the index for where to insert
                    enumerate(

                        # Insert the gene between chromosome[i-1] and chromosome[i]
                        # Edge cases occur at the endpoints of the path
                        sum((
                            self.dist(gene, chromosome[i])   if i < len(chromosome) else 0,
                            self.dist(gene, chromosome[i-1]) if i + self.cycle > 0  else 0,
                           -self.dist(chromosome[i], chromosome[i-1]) if (0<i<len(chromosome) or self.cycle == 1) else 0
                        ))
                        for i
                        in range(len(chromosome)+1-self.cycle)
                    ),

                    # Minimize by distance
                    key = lambda elem: elem[1]

                # Return index
                )[0],

                # Return gene
                gene

            # End chromosome.insert
            )

        # Update fitness
        chromosome.fitness = self.fitness_function_impl(chromosome)

        return chromosome

    #===================#
    # Custom GA Methods #
    #===================#

    def crossover_individual_impl(self, parent_1, parent_2):
        """Crosses two parents by keeping parts of the paths that they share
        and then greedily inserting the remaining points back in."""

        child = self.make_chromosome([])
        j = None

        # Path goes full cycle, so endpoints can be checked
        if self.cycle == 1:
            removed_genes = []

        # Path does not go full cycle, so endpoints cannot be checked
        else:
            removed_genes = [parent_1[0], parent_1[-1]]

        # Check each gene from parent 1 in reversed order
        for i in range((2, 0)[self.cycle], len(parent_1)):

            # Get the 2 surrounding genes in a row from parent 1
            genes_1 = [parent_1[i-2], parent_1[i]]

            # Search for the middle gene from parent 1 in parent 2
            j = parent_2.index(parent_1[i-1], j)

            # Get the 2 surrounding genes in a row from parent 2
            genes_2 = [parent_2[j-2], parent_2[j]]

            # If the genes match
            if all(gene in genes_1 for gene in genes_2) and (self.cycle == 1 or j > 1):
                child.append(parent_1[i-1])

            # If genes don't match, remove them
            else:
                removed_genes.append(parent_1[i-1])

        # Insert all of the removed points
        self.population.add_child(self.greedy_insertion(child, removed_genes))


    def mutation_individual_impl(self, chromosome):
        """Mutate a chromosome by randomly removing sqrt(n) genes and re-inserting them."""

        removed_genes = [
            chromosome.pop(index)
            for index
            in sorted(
                random.sample(
                    range(len(chromosome)),
                    int(np.sqrt(len(chromosome))+1)
                ),
                reverse = True
            )
        ]

        # Reinsert genes
        self.greedy_insertion(chromosome, removed_genes)


    def adapt_population(self):
        """Adapt the population by applying 2-opt to the best chromosome,
        and then attempt to greedily re-insert every chromosome one by one,
        and then search for the best point to split the cycle (if wanted)."""

        # Modify the best chromosome
        chromosome = self.population[0]

        # Try all combinations to reverse, assuming i < j.
        for i in range(1-self.cycle, len(chromosome)):
            for j in range(i+2, len(chromosome)):

                # Compute distances with and without reversing a segment
                original_dist = self.dist(chromosome[i], chromosome[i-1]) + self.dist(chromosome[j], chromosome[j-1])
                reversed_dist = self.dist(chromosome[i], chromosome[j]) + self.dist(chromosome[i-1], chromosome[j-1])

                # Apply possible optimizations
                if original_dist > reversed_dist:
                    chromosome[i:j] = reversed(chromosome[i:j])

        # Greedily re-insert every gene
        for i, gene in enumerate(chromosome):
            chromosome.pop(i)
            self.greedy_insertion(chromosome, [gene])

        # Find the best spot to split the full cycle if a full cycle is not wanted
        if self.cycle == 0:

            # Search by maximum value
            i = max(

                # Track the index for splitting at i, i-1
                enumerate(

                    # Distance of splitting point
                    self.dist(chromosome[i], chromosome[i-1])
                    for i
                    in range(len(chromosome))
                ),

                # Maximize by distance
                key = lambda elem: elem[1]

            # Return by index
            )[0]

            chromosome[:] = chromosome[i:] + chromosome[:i]


    def termination_impl(self):
        """Terminate if the maximum number of generations is reached or no change has occured in a while."""

        # Update best fitness/generation
        try:
            if any((all((self.target_fitness_type == 'min',
                         self.population[0].fitness < self.best_chromosome.fitness * (1-1e-8))),
                    all((self.target_fitness_type == 'max',
                         self.population[0].fitness > self.best_chromosome.fitness * (1+1e-8))),
                )):
                self.best_chromosome = self.make_chromosome(self.population[0])
                self.best_generation = self.current_generation
                self.best_chromosome.fitness = self.population[0].fitness

        # No population or fitness yet
        except (TypeError, AttributeError):
            if self.population is not None:
                self.best_chromosome = self.make_chromosome(self.population[0])
                self.best_chromosome.fitness = self.population[0].fitness

        # Check for termination
        return self.current_generation < min(self.generation_goal, self.best_generation + self.change_goal)


# If this file is being run directly, show an example using TSP_Solver()
if __name__ == '__main__':

    from matplotlib import animation, rc
    rc('animation', html='html5')

    print("""Recommended settings:
    2 dimensions
    100 points
    10 chromosomes
    True full cycle
    0.1 adapt rate (once every 10 generations)
    15 generations without change
    100 generations run at most
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
                cycle_flag = 1
            elif cycle_flag in ("False", "false"):
                cycle_flag = 0
            else:
                raise ValueError("Invalid, input either True or False.")

            if not (0 <= (adapt_rate := float(input("Adapt rate : "))) < 1):
                raise ValueError("Invalid, input a float between 0 and 1. Recommended 0.02 to adapt every 50th generation.")

            if (max_no_change := int(input("Number of generations without change before stopping : "))) < 2:
                raise ValueError("Invalid, input an integer greater than or equal to 2.")

            if (max_generation := int(input("Maximum number of generations run : "))) < 2:
                raise ValueError("Invalid, input an integer greater than or equal to 1.")

            if dimensions in (2, 3):
                file_name = input("Name of gif file (without the .gif) : ")

            # Successful inputs
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

    solver = TSP_Solver(
        data,
        cycle_flag,
        population_size = population_size,
        adapt_rate = adapt_rate,
        max_generation = max_generation,
        max_no_change = max_no_change
    )

    # Make figure, plot, and line
    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(projection = '3d' if dimensions == 3 else None)
    line, = ax.plot([], [], 'bo-')

    # Set axis limits to (0, 1)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    if dimensions == 3:
        ax.set_zlim((0, 1))

    # Store the paths
    X = []

    def init():
        line.set_data(*([[]]*dimensions))
        return (line,)

    def animate(i):
        line.set_xdata(X[i][0])
        line.set_ydata(X[i][1])
        if dimensions == 3:
            line.set_3d_properties(X[i][2])
        path_len = sum(
            np.sqrt(sum(
                (X[i][j][k] - X[i][j][k-1]) ** 2
                for j
                in range(dimensions)
            ))
            for k
            in range(1, len(X[i][0]))
        )
        plt.title(f"Generation: {max(0, min(solver.current_generation, i-10))}\nPath Length: {round(path_len, 5)}")
        return (line,)

    while solver.active():

        # Evolve 1 generation
        solver.evolve(1)

        # Show best chromosome
        solver.print_generation()
        solver.print_best_chromosome()
        print()

        X.append([
            [
                data[solver.population[0][index].value][component]
                for index
                in range(-cycle_flag, len(solver.population[0]))
            ]
            for component
            in range(dimensions)
        ])


    # Print results
    solver.print_generation()
    solver.print_population()

    # Print data
    print("Data points:")
    for gene in solver.best_chromosome:
        print(data[gene.value])

    # Print settings
    print(f"""
    Settings:
    {dimensions} dimensions
    {number_of_points} points
    {population_size} chromosomes
    A full cycle was{"" if cycle_flag == 1 else "n't "} used
    {adapt_rate} adapt rate (once every {round(1/adapt_rate, 2) if adapt_rate>0 else float('inf')} generations)
    {max_no_change} generations without change
    {max_generation} generations run at most
    {file_name}.gif is your gif file.
    """)

    # Animate plot
    if dimensions in (2, 3):
        X = [X[0]] * 9 + X + [X[-1]] * 9
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=(solver.current_generation+18), interval=20, blit=True)
        anim.save(f'{file_name}.gif', writer='imagemagick', fps=5)
