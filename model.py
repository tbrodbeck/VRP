# Libraries Included:
# Numpy, Scipy, Scikit, Pandas

import numpy as np
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from AntColonyOptimizer import AntColonyOptimizer
from copy import deepcopy

vrp1_dir = "VRP1/"
vrp2_dir = "VRP2/"


def load_vrp(which):
    """
    load scenario information
    """
    which = int(which)
    print('Scenario:', which)
    if which == 1:
        capacity = np.loadtxt(vrp1_dir + "capacity.txt")
        demand = np.loadtxt(vrp1_dir + "demand.txt")
        distance = np.loadtxt(vrp1_dir + "distance.txt")
        transportation_cost = np.loadtxt(vrp1_dir + "transportation_cost.txt")
        return capacity, demand, distance, transportation_cost
    elif which == 2:
        capacity = np.loadtxt(vrp2_dir + "capacity.txt")
        demand = np.loadtxt(vrp2_dir + "demand.txt")
        distance = np.loadtxt(vrp2_dir + "distance.txt")
        transportation_cost = np.loadtxt(vrp2_dir + "transportation_cost.txt")
        return capacity, demand, distance, transportation_cost
    else:
        raise ValueError("Invalid VRP problem number!")


class Fitness:
    """
    Determines the fitness value of a chromosome using ACO
    """

    def __init__(self, capacity, demand, distance_matrix, transportation_cost,
                 iterations, beta, evap_rate, beta_evap):  # move ant colony parameters here?
        self.capacity = capacity
        self.demand = demand
        self.distance_matrix = distance_matrix
        self.transportation_cost = transportation_cost
        self.iterations = iterations
        self.beta = beta
        self.evap_rate = evap_rate
        self.beta_evap = beta_evap

    def fit(self, chromosome):
        """
        Sum up the costs for all the shortest routes aco was able to find
        """

        cost = 0

        for i in range(len(self.transportation_cost)):

            acoDistance = deepcopy(self.distance_matrix)
            k = 0
            load = 0

            """
            At first we slice the customers who are not members of the subgraph out of the matrice
            """

            for j in range(len(chromosome)):

                if not (chromosome[j] == i):
                    acoDistance = np.delete(acoDistance, j - k + 1, 0)
                    acoDistance = np.delete(acoDistance, j - k + 1, 1)
                    k += 1
                else:
                    # if the capacity constraint is violated, False shall be returned
                    load += self.demand[j]
                    if load > self.capacity[i]:
                        print("False Fitness")
                        return False

            # Sum up costs
            aco = AntColonyOptimizer(ants=10, evaporation_rate=self.evap_rate, intensification=2, alpha=1, beta=self.beta,beta_evaporation_rate=self.beta_evap, choose_best=.1)
            current = aco.fit(acoDistance, self.iterations)
            cost += min(aco.best) * self.transportation_cost[i]

        return cost


class Chromosome:
    """
    Helper functions for the handling of chromosomes
    """

    def __init__(self, allele, fitness_object, available_capacity):
        self.allele = allele
        self.fitness = fitness_object.fit(allele)  # determine fitness value of a chromosome during instantiation
        self.available_capacity = available_capacity  # for every vehicle store the not used capacity

    def refresh_fitness(self, fitness_object):
        """
        Updates the fitness value of a chromosome after mutation/recombination
        """

        self.fitness = fitness_object.fit(self.allele)

    def equals(self, other):
        """
        Returns true if 2 Chromosomes have the same allels
        """

        if len(self.allele) != len(other.allele):
            return False

        answer = True

        for i in range(len(self.allele)):
            if self.allele[i] != other.allele[i]:
                answer = False

        return answer


class Initializer:
    """
    Intitializes a population
    """

    def __init__(self, capacity, demand, fitness_object):
        self.capacity = capacity
        self.num_vehicles = len(capacity)
        self.demand = demand
        self.num_customers = len(demand)
        self.population = []
        self.fitness_object = fitness_object

    def initialize(self, pop_size):
        """
        initializes a population of size "pop_size"
        """
        pass


class KMeansInitializer(Initializer):
    """
    Initializes a population based on the k-means clustering algorithm
    """

    def __init__(self, capacity, demand, fitness_object, reduce_clusters=0, num_iterations=0, squared_dist=True):

        self.capacity = capacity
        self.demand = demand
        self.num_customers = len(demand)
        self.fitness_object = fitness_object
        self.population = []
        self.distance_matrix = fitness_object.distance_matrix
        self.squared_dist = squared_dist

        if reduce_clusters == 0:
            # default = all vehicles will have a cluster assigned
            self.num_clusters = len(capacity)
        else:
            # number of vehicles being used
            self.num_clusters = int(len(capacity) - reduce_clusters)

        if num_iterations == 0:
            self.num_iterations = 20
        else:
            self.num_iterations = num_iterations

    def initialize(self, pop_size):
        """
        Clusters the customers according to their shortest distances and creates chromosomes from these clusters
        """

        for i in range(pop_size):
            
            valid = False
            while not(valid):
                
                chromosome = np.zeros(self.num_customers, dtype=int)
    
                # number of kmeans iterations
                for iteration in range(self.num_iterations):
    
                    if iteration == 0:
                        # Choose starting customers by random
    
                        for i in range(self.num_clusters):
                            customer = np.random.randint(self.num_customers)
    
                            if chromosome[customer] == 0:
    
                                chromosome[customer] = i + 1
    
                            else:
                                while chromosome[customer] != 0:
                                    customer = (customer + 1) % self.num_customers
    
                                chromosome[customer] = i + 1
    
                    else:
                        # set new starting points from clusters
                        # starting point will be the one customer whose squared distance to all others is smallest
                        distance = np.zeros(self.num_clusters, dtype=int)
                        position = np.zeros(self.num_clusters, dtype=int)
    
                        for i in range(self.num_customers):
                            cluster_size = 0
                            summed_distance = 0
    
                            # sum up distances
                            for j in range(self.num_customers):
                                if chromosome[i] == chromosome[j] and i != j:
                                    # squared distance
                                    summed_distance += self.fitness_object.distance_matrix[i + 1][j + 1] ** 2
                                    cluster_size += 1
    
                            # update best
                            if cluster_size > 0:
                                if distance[chromosome[i] - 1] == 0:
                                    distance[chromosome[i] - 1] = summed_distance / cluster_size
                                    position[chromosome[i] - 1] = i
                                else:
                                    if summed_distance / cluster_size < distance[chromosome[i] - 1]:
                                        distance[chromosome[i] - 1] = summed_distance / cluster_size
                                        position[chromosome[i] - 1] = i
    
                        for i in range(self.num_customers):
                            chromosome[i] = 0
                        for i in range(self.num_clusters):
                            chromosome[position[i]] = i + 1
    
                    # cluster assignment
                    # choose an unassigned customer
                    for i in range(self.num_customers - self.num_clusters):
                        customer = np.random.randint(self.num_customers)
    
                        if chromosome[customer] != 0:
                            while chromosome[customer] != 0:
                                customer = (customer + 1) % self.num_customers
    
                        cluster_distance = np.zeros(self.num_clusters, dtype=int)
                        cluster_size = np.zeros(self.num_clusters, dtype=int)
                        cluster_load = np.zeros(self.num_clusters, dtype=int)
    
                        # calculate distance to clusters
                        for i in range(self.num_customers):
                            if chromosome[i] != 0:
                                if self.squared_dist:
                                    cluster_distance[chromosome[i] - 1] += self.fitness_object.distance_matrix[i + 1][customer + 1] ** 2
                                else:
                                    cluster_distance[chromosome[i] - 1] += self.fitness_object.distance_matrix[i + 1][customer + 1]
                                cluster_size[chromosome[i] - 1] += 1
                                cluster_load[chromosome[i] - 1] += self.demand[i]
    
                        for i in range(self.num_clusters):
                            if cluster_size[i] > 0:
                                cluster_distance[i] /= cluster_size[i]
    
                        # lowest distance first !capacity constraint
                        # cluster_sorted = np.sort(deepcopy(cluster_distance)) #?
                        index = np.argsort(cluster_distance)
    
                        done = False
                        i = 0
                        while i < self.num_clusters and not (done):
                            cluster_load[index[i]] += self.demand[customer]
                            if self.capacity_constraint(cluster_load):
                                chromosome[customer] = index[i] + 1
                                done = True
                            else:
                                cluster_load[index[i]] -= self.demand[customer]
                                i += 1
                                # exit statement if no solution found? possible?
    
                # assign each cluster a vehicle
                allele = self.assign_vehicle(cluster_load, chromosome)
                available_capacity = deepcopy(self.capacity)
    
                # construct chromosome
                for i in range(self.num_customers):
                    available_capacity[allele[i]] -= self.demand[i]
                    # control statement?
                individual = Chromosome(allele, self.fitness_object, available_capacity)
                
                
                # insert chromosome into population
                if individual.fitness != False:
                    valid=True
                    self.population.append(individual)

        return self.population


    def capacity_constraint(self, cluster_load):
        """
        Checks whether there is a valid vehicle assignment for the clusters.
        Return False if the capacity constraint is violated.
        """
        clusters = np.sort(cluster_load)
        difference = len(self.capacity) - len(cluster_load)

        for i in range(len(cluster_load)):
            if self.capacity[i + difference] < clusters[i]:
                return False

        return True

    def assign_vehicle(self, cluster_load, chromosome):
        """
        Assigns every cluster a vehicle
        """
        order = np.argsort(cluster_load)
        cluster_vehicles = np.zeros(self.num_clusters, dtype=int)

        assigned = 0
        vehicle = 0

        while assigned < self.num_clusters and vehicle < len(self.capacity):

            # assign each vehicle the cluster with the currently highest load the vehicle can carry
            cluster = 0
            while self.capacity[vehicle] > cluster_load[order[cluster]] and cluster < len(order) - 1:
                cluster += 1

            if cluster > 0:
                cluster_vehicles[order[cluster - 1]] = vehicle
                assigned += 1
                if len(order) > 0:
                    order = np.delete(order, cluster - 1)
            elif cluster == len(order) - 1:
                cluster_vehicles[order[cluster]] = vehicle
                assigned += 1
            else:
                print("No vehicle match.")
                return False

            vehicle += 1
            # control statement?

        allele = []
        for i in range(self.num_customers):
            allele.append(cluster_vehicles[chromosome[i] - 1])
        return allele


class ConstraintRandomInitializer(Initializer):
    """
    Random initialization of chromosomes
    """

    def initialize(self, pop_size):
        """
        Generates a population of Chromosomes with random alleles with regard
        to the capacity constraint
        """

        for i in range(pop_size):

            init = []

            available_capacity = deepcopy(self.capacity)
            for j in range(self.num_customers):
                # generate random vehicle_number and insert it in the chromosome if it doesn't violate capacity constraint

                vehicle_number = np.random.randint(self.num_vehicles)

                while self.demand[j] > available_capacity[vehicle_number]:
                    vehicle_number = np.random.randint(self.num_vehicles)

                init.append(vehicle_number)
                available_capacity[vehicle_number] -= self.demand[j]

            chromosome = Chromosome(init, self.fitness_object, available_capacity)
            self.population.append(chromosome)

        return self.population


class Selector():
    """
    Selects Candidates
    """

    def __init__(self, selection_size):
        self.size = selection_size

    @abstractmethod
    def select(self, population):
        pass


class TournamentSelector(Selector):
    """
    Selects the fittest individuals from the population
    """

    def __init__(self, selection_size, tournament_size=2):

        self.selection_size = selection_size
        self.tournament_size = tournament_size

    def select(self, population):
        """
        Choose Chromosomes from the population randomly and put them into lists (tournaments)
        The best individual in regard to its fitness value will be added to the selection list
        """

        if len(population) < self.selection_size * self.tournament_size:
            return False

        self.mating_pool = []
        old_population = deepcopy(population)

        for i in range(self.selection_size):

            tournament = []
            # Choose random individuals
            for j in range(self.tournament_size):
                tournament.append(old_population.pop(np.random.randint(len(old_population))))

            # Use best
            self.mating_pool.append(min(tournament, key=lambda Chromosome: Chromosome.fitness))

        return self.mating_pool


class Recombiner:
    """
    The recombiner performs crossover between 2 individuals of the mating-pool
    """

    def __init__(self, crossover_prob, fitness_obj, demand):
        self.crossover_prob = crossover_prob
        self.fitness_obj = fitness_obj
        self.demand = demand

    def recombine(self, mating_pool):
        pass


class UniformRecombiner(Recombiner):
    """
    Performs crossover at every allele with a certain probability (crossover_prob)
    """

    def recombine(self, mating_pool):
        """
        Recombines to chromosomes at every allele with a certain probability
        Checks for the capacity constraint and updates the fitness value if necessary
        """

        self.offspring = []

        while len(mating_pool) > 0:

            change = False
            parent_one = mating_pool.pop(np.random.randint(len(mating_pool)))
            parent_two = mating_pool.pop(np.random.randint(len(mating_pool)))

            for i in range(len(self.demand)):
                # Would a crossover make a difference?
                if not (parent_one.allele[i] == parent_two.allele[i]):
                    # would an exchange of chromosomes lead to a new valid chromosome? (capacity constraint)
                    if parent_one.available_capacity[parent_two.allele[i]] > self.demand[i] and \
                            parent_two.available_capacity[parent_one.allele[i]] > self.demand[i]:
                        # crossover probability
                        if np.random.random() < self.crossover_prob:
                            change = True
                            parent_one.available_capacity[parent_one.allele[i]] += self.demand[i]
                            parent_one.available_capacity[parent_two.allele[i]] -= self.demand[i]
                            parent_two.available_capacity[parent_two.allele[i]] += self.demand[i]
                            parent_two.available_capacity[parent_one.allele[i]] -= self.demand[i]

                            buffer = parent_one.allele[i]
                            parent_one.allele[i] = parent_two.allele[i]
                            parent_two.allele[i] = buffer

            # Actualize the fitness value of the new Chromosome if there was a change
            if change:
                parent_one.refresh_fitness(self.fitness_obj)
                parent_two.refresh_fitness(self.fitness_obj)

            self.offspring.append(parent_one)
            self.offspring.append(parent_two)

        return self.offspring


class Mutator:
    """
    The Mutator mutates alleles with a certain probability to add diversity to the population
    """

    def __init__(self, mutation_prob, fitness_obj, demand, vehicle_number):
        self.mutation_prob = mutation_prob
        self.fitness_obj = fitness_obj
        self.demand = demand
        self.vehicle_number = vehicle_number
        self.chromosome_length = len(self.demand)

    def mutate(self, offspring):
        pass

    def steady_mutate(self, offspring):
        """
        Mutates the individuals in the offspring but also keeps the previous alleles.
        Therefore mutation cannot lower the fitness value of the offspring.
        """
        self.offspring = []

        for i in range(len(offspring)):
            self.offspring.append(deepcopy(offspring[i]))

        self.mutate(self.offspring)

        for i in range(len(self.offspring)):
            offspring.append(self.offspring[i])


class BitFlipMutator(Mutator):
    """
    Changes single alleles of a chromosome
    """

    def mutate(self, offspring):
        """
        Mutates alleles with a certain probability by random (mutation_prob) with respect to capacity constraints
        """

        for i in range(len(offspring)):
            change = False
            for j in range(self.chromosome_length):
                if np.random.random() < self.mutation_prob:

                    current = offspring[i].allele[j]
                    options = []
                    # Is there a mutation that leads to a new valid chromosome?
                    for k in range(max(offspring[i].allele)):
                        if not (k == current) and offspring[i].available_capacity[k] > self.demand[j]:
                            options.append(k)

                    if options:
                        change = True

                        # determine a new vehicle number that fulfills the capacity constraint
                        # number = np.random.randint(len(options))
                        new = options[np.random.randint(len(options))]

                        # update the available capacity and the allel afterwards
                        offspring[i].available_capacity[current] += self.demand[j]
                        offspring[i].available_capacity[new] -= self.demand[j]
                        offspring[i].allele[j] = new

            if change:
                offspring[i].refresh_fitness(self.fitness_obj)


class Replacer:
    """
    Decides which chromosomes to keep for next generation
    """

    def __init__(self, n, initializer_obj):
        self.n = n
        self.initializer = initializer_obj

    def replace(self, population, offspring):
        pass


class SteadyStateReplacer(Replacer):
    """
    Creates the next generation of the population
    Takes the n best chromosomes from the offspring and the initial population and initializes new ones
    """

    def replace(self, population, offspring):
        """
        Keeps the best n individuals from population and offspring for the next generation.
        The remaining slots are refilled with the Initializer.
        """

        self.next_gen = []
        pop_size = len(population)
        population = sorted(population, key=lambda Chromosome: Chromosome.fitness)
        offspring = sorted(offspring, key=lambda Chromosome: Chromosome.fitness)

        for i in range(self.n):
            # if there are individuals left in the offspring
            if offspring:
                if population:
                    if offspring[0].fitness < population[0].fitness:
                        self.next_gen.append(offspring.pop(0))
                    else:
                        self.next_gen.append(population.pop(0))
                else:
                    self.next_gen.append(offspring.pop(0))
            # when offspring is empty just add the individuals with the best fitness values from old population
            else:
                if population:
                    self.next_gen.append(population.pop(0))
                else:
                    pass


class SteadyStateNoDuplicatesReplacer(Replacer):
    """
    Creates the next generation of the population
    Takes the n best chromosomes from the offspring and the initial population and initializes new ones
    Does not use duplicates and invalid chromosomes
    """


    def replace(self, population, offspring):
        """
        Keeps the best n individuals from population and rest from offspring for the next generation.
        Neither duplicates nor Chromosomes with invalid solutions (.fitness == false) will occur in next_gen.
        """

        # set up next generation and sort population & offspring by their fitness value
        self.next_gen = []
        pop_size = len(population)
        population = sorted(population, key=lambda Chromosome: Chromosome.fitness)
        offspring = sorted(offspring, key=lambda Chromosome: Chromosome.fitness)

        # keep n individuals
        for i in range(self.n):
            # If there are individuals left in the current population
            # check for duplicates
            if i == 0:
                while population[0].fitness == False:
                    population.pop(0)
            j = 0
            while j < i and population:
                if population[0].equals(self.next_gen[j]) or population[0].fitness == False:
                    population.pop(0)
                    j = 0
                else:
                    j += 1

            if population:
                self.next_gen.append(population.pop(0))

        i = 0
        # add rest
        while i < (pop_size - self.n):
            unique = True
            for j in range(len(self.next_gen)):
                if offspring:
                    if self.next_gen[j].equals(offspring[0]) or offspring[0].fitness == False:
                        offspring.pop(0)
                        unique = False
                if population:
                    if self.next_gen[j].equals(population[0]) or population[0].fitness == False:
                        population.pop(0)
                        unique = False
            if unique:
                if offspring:
                    if population:
                        if offspring[0].fitness < population[0].fitness:
                            self.next_gen.append(offspring.pop(0))
                            i += 1
                        else:
                            self.next_gen.append(population.pop(0))
                            i += 1
                    else:
                        self.next_gen.append(offspring.pop(0))
                        i += 1
                else:
                    if population:
                        self.next_gen.append(population.pop(0))
                        i += 1
                    else:
                        i += 1

        return self.next_gen


def VRP(scenario, heuristic, pop_size, selection_size, aco_iterations, beta, evap_rate, beta_evap, crossover_prob, mutation_prob, reduce_clusters, kmeans_iterations, squared_dist, time_limit, verbose=False):
    """
    This is our solver for the vehicle routing problem.

    :param scenario: takes scenario [1,2] of the problem
    :param pop_size: population size of our Genetic Algorithm (GA)
    :param selection_size: amount of chromosomes that get selected by the selector of the GA
    :param crossover_prob: probability parameter of the recombiner of the GA
    :param mutation_prob: probability parameter of the mutator of the GA
    :param aco_iterations: amount of Ant Colony Optimizations we perform
    :param reduce_clusters: amount of clusters of the heuristic-initializer of the GA
    :param kmeans_iterations: amount of heuristic iterations of the heuristic-initializer
    :param time_limit: time-limit for the algorithm deployment
    :return:
    """

    # convert inputs to right values
    pop_size = int(pop_size)
    aco_iterations = int(aco_iterations)
    kmeans_iterations = int(kmeans_iterations)
    selection_size = int(selection_size)

    tournament_size = 2
    steady_state_n = round(pop_size / 2)

    capacity, demand, distance, transportation_cost = load_vrp(scenario)
    print("solve VRP with nr of costumers: %s, nr of vehicles: %d" % (len(demand), len(capacity)))

    fitness = Fitness(capacity, demand, distance, transportation_cost, aco_iterations, beta, evap_rate, beta_evap)

    # selecting initializer
    if heuristic:
        initializer = KMeansInitializer(capacity, demand, fitness, reduce_clusters, kmeans_iterations, squared_dist)
    else:
        initializer = ConstraintRandomInitializer(capacity, demand, fitness)

    population = initializer.initialize(pop_size)

    print("Initial Population:")
    print([chromo.fitness for chromo in population])

    current_time = time.time()
    best = []
    mean = []
    solution = None

    while time.time() - current_time < time_limit:

        selector = TournamentSelector(selection_size, tournament_size)
        mating_pool = selector.select(population)

        if verbose:
            print("Mating Pool:")
            for i in range(len(mating_pool)):
                print(mating_pool[i].fitness)

        recombiner = UniformRecombiner(crossover_prob, fitness, demand)
        offspring = recombiner.recombine(mating_pool)

        if verbose:
            print("Offspring:")
            for i in range(len(offspring)):
                print(offspring[i].fitness)

        mutator = BitFlipMutator(mutation_prob, fitness, demand, len(capacity))
        # mutator.mutate(offspring)
        mutator.steady_mutate(offspring)

        if verbose:
            print("Mutated Offspring")
            for i in range(len(offspring)):
                print(offspring[i].fitness)

        # replacer = SteadyStateReplacer(steady_state_n, initializer)
        replacer = SteadyStateNoDuplicatesReplacer(steady_state_n, initializer)
        population = replacer.replace(population, offspring)

        if verbose:
            print("New Population:")
            for i in range(len(population)):
                print(population[i].fitness)

        best_pop = 0
        mean_pop = 0
        for i in range(len(population)):
            if best_pop == 0 or best_pop > population[i].fitness:
                best_pop = population[i].fitness
                solution = population[i].allele
            mean_pop += population[i].fitness
        mean_pop /= len(population)
        best.append(best_pop)
        mean.append(mean_pop)

    print('Best:', best)
    print('Mean:', mean)
    print('Solution:', solution)


    return best, mean, solution

if __name__ == '__main__':
    # VRP(scenario, heuristic, pop_size, selection_size, aco_iterations, beta, evap_rate, beta_evap, crossover_prob, mutation_prob, reduce_clusters, kmeans_iterations, squared_dist, time_limit)
    VRP(2, True, 12, 2, 10, 1, 0.1, 0, 0.2, 0.1, 0, 10, True, 60)
