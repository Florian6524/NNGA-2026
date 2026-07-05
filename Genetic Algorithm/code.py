#the code was done in google colab, in different cells, i will separate the cells by comments

#cell 1, data:

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

data = {
    'Vendor': ['1. Pepco Orhidea', '2. Carrefour Orhidea', '3. Pepco Afi', '4. Auchan Afi', '5. Lego Afi', '6. Lego Promenada', '7. Noriel Promenada', '8. Lego ParkLake', '9. Carturesti ParkLake', '10. Pepco ParkLake', '11. Lego Baneasa', '12. Carturesti Baneasa', '13. Carrefour Hypermarket', '14. Lidl Pucheni', '15. Pepco Pucheni', '16. Atac Hiper Discount Auchan', '17. Lidl Soseaua Banatului', '18. Penny Bacanie', '19. Lidl Bacanie', '20. Selgros Pantelimon', '21. Kaufland Pantelimon', '22. Jumbo Berceni', '23. Jumbo China', '24. Jumbo Militari', '25. Metro Voluntari'],
    'Lat': [44.445948463317166, 44.44473164621837, 44.431580575111866, 44.43101117158002, 44.431296580183655, 44.478139259693286, 44.47808450654406, 44.41997043603905, 44.42090430099619, 44.42140046007508, 44.509127236890784, 44.506793575294, 44.37420445066149, 44.38623091897692, 44.38613226356797, 44.49793369143791, 44.50457403726575, 44.54154323850486, 44.53928140925567, 44.45320795589227, 44.4505629595182, 44.36460022332874, 44.40680122380867, 44.440393043211614, 44.49668554403672],
    'Lon': [26.06353865914826, 26.063296355185372, 26.05083120713418, 26.049975484268163, 26.05235077116358, 26.10339111589151, 26.10400399563018, 26.149771754988734, 26.14937670530996, 26.14999093116092, 26.088556570573388, 26.089918986689497, 26.12046654656173, 26.058959406839048, 26.059980529982763, 26.000214297802202, 25.989371792281645, 26.14286168772606,  26.14207790662546, 26.222119862958607, 26.21860219976802, 26.12242923141029, 26.061317778233143, 25.95488772074926, 26.19864667539159]
}
df = pd.DataFrame(data)
num_vendors = len(df)|

#cell 2, distance function

def get_distance_matrix(df):
    coords = df[['Lat', 'Lon']].values
    matrix = np.zeros((num_vendors, num_vendors))
    for i in range(num_vendors):
        for j in range(num_vendors):
            matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))
    return matrix

dist_matrix = get_distance_matrix(df)

#cell 3, route, crossover, mutate

def get_route_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i], route[i+1]]
    distance += dist_matrix[route[-1], route[0]]
    return distance

def calculate_fitness(route):
    return 1 / get_route_distance(route)

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(num_vendors), 2))
    child = [None] * num_vendors
    child[start:end] = parent1[start:end]

    pointer = 0
    for item in parent2:
        if item not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = item
    return child

def mutate(route, rate=0.05):
    if random.random() < rate:
        i, j = random.sample(range(num_vendors), 2)
        route[i], route[j] = route[j], route[i]
    return route

#cell 4, first traversal

pop_size = 150
generations = 1000
stagnation_limit = 50
mutation_rate = 0.07

population = [random.sample(range(num_vendors), num_vendors) for _ in range(pop_size)]
history = []
best_dist_overall = float('inf')
gens_without_improvement = 0

print(f"Starting Standard Evolution (Distance Optimization)")
print(f"{'Gen':<5} | {'Best Individual (Route)':<60} | {'Distance':<10}")
print("-" * 95)

for gen in range(generations):
    population = sorted(population, key=lambda x: calculate_fitness(x), reverse=True)

    current_best_route = population[0]
    current_best_dist = get_route_distance(current_best_route)
    history.append(current_best_dist)

    if current_best_dist < best_dist_overall:
        best_dist_overall = current_best_dist
        gens_without_improvement = 0
    else:
        gens_without_improvement += 1

    if gen % 10 == 0:
        print(f"{gen:<5} | {str(current_best_route):<60} | {current_best_dist:.6f}")

    if gens_without_improvement >= stagnation_limit:
        print("-" * 95)
        print(f"EARLY STOPPING: No improvement for {stagnation_limit} generations.")
        print(f"Final Convergence reached at Generation {gen}.")
        break

    new_gen = [population[0], population[1]]

    while len(new_gen) < pop_size:
        p1, p2 = random.sample(population[:pop_size // 2], 2)
        child = crossover(p1, p2)
        child = mutate(child, rate=mutation_rate)
        new_gen.append(child)

    population = new_gen

best_route = population[0]
print("-" * 95)
print(f"Final Best Distance: {get_route_distance(best_route):.6f}")

#cell 5, plot the best route

def plot_final_route(df, route):
    ordered_df = df.iloc[route + [route[0]]]

    plt.figure(figsize=(15, 10))

    plt.plot(ordered_df['Lon'], ordered_df['Lat'], 'o-', mfc='red', markersize=8, linewidth=2, color='royalblue', label='Path')

    plt.plot(df.iloc[route[0]]['Lon'], df.iloc[route[0]]['Lat'], 'rs', markersize=12, label='Start/End')

    for i, name in enumerate(df['Vendor']):
        plt.annotate(name, (df['Lon'][i], df['Lat'][i]), xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.title("Optimized Lego Vendor Route (Bucharest)", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

plot_final_route(df, best_route)

#cell 6, plot the convergence

def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history, color='darkorange', linewidth=2)
    plt.title("Genetic Algorithm Convergence", fontsize=14)
    plt.xlabel("Generation")
    plt.ylabel("Total Distance (Degrees)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.annotate(f'Start: {history[0]:.4f}', (0, history[0]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f'Final: {history[-1]:.4f}', (len(history)-1, history[-1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.show()

plot_history(history)

#cell 7, traversal based on importance, official lego stores first

START_INDEX = 0 #change index here to choose starting shop

lego_indices = [i for i, name in enumerate(df['Vendor']) if 'Lego' in name]

print(f"Starting at: {df.iloc[START_INDEX]['Vendor']}")
print(f"Priority Stores identified at indices: {lego_indices}")

def get_weighted_fitness(route):
    dist = get_route_distance(route)
    lego_penalty = 0
    for position, vendor_idx in enumerate(route):
        if vendor_idx in lego_indices:
            lego_penalty += (position**2) * 0.005

    return 1 / (dist + lego_penalty)

#cell 8, new mutate and traverse

  def constrained_mutate(route, rate=0.07):
    if random.random() < rate:
        i, j = random.sample(range(1, num_vendors), 2)
        route[i], route[j] = route[j], route[i]
    return route

pop_size = 150
generations = 500
stagnation_limit = 50
mutation_rate = 0.1

best_dist_overall = float('inf')
gens_without_improvement = 0
history_weighted = []

population = []
for _ in range(pop_size):
    remaining = [i for i in range(num_vendors) if i != START_INDEX]
    random.shuffle(remaining)
    population.append([START_INDEX] + remaining)

print(f"Starting Priority Evolution (Start: {df.iloc[START_INDEX]['Vendor']})")
print(f"{'Gen':<5} | {'Best Individual (Route)':<60} | {'Distance':<10}")
print("-" * 95)

for gen in range(generations):
    population = sorted(population, key=lambda x: get_weighted_fitness(x), reverse=True)

    best_current_route = population[0]
    current_dist = get_route_distance(best_current_route)
    history_weighted.append(current_dist)

    if current_dist < best_dist_overall:
        best_dist_overall = current_dist
        gens_without_improvement = 0
    else:
        gens_without_improvement += 1

    if gen % 10 == 0:
        print(f"{gen:<5} | {str(best_current_route):<60} | {current_dist:.4f}")

    if gens_without_improvement >= stagnation_limit:
        print("-" * 95)
        print(f"EARLY STOPPING: No improvement for {stagnation_limit} generations.")
        print(f"Final Convergence reached at Generation {gen}.")
        break

    new_gen = [population[0], population[1]]

    while len(new_gen) < pop_size:
        p1, p2 = random.sample(population[:pop_size//2], 2)

        child = crossover(p1, p2)

        if child[0] != START_INDEX:
            s_pos = child.index(START_INDEX)
            child[0], child[s_pos] = child[s_pos], child[0]

        child = constrained_mutate(child, rate=mutation_rate)

        new_gen.append(child)

    population = new_gen

best_priority_route = population[0]
print("-" * 95)
print(f"Final Priority Route: {best_priority_route}")

#cell 9, plot the route

def plot_priority_route(df, route):
    ordered_df = df.iloc[route + [route[0]]]
    plt.figure(figsize=(15, 10))

    plt.plot(ordered_df['Lon'], ordered_df['Lat'], 'o-', mfc='orange', linewidth=2, color='green')

    plt.plot(df.iloc[route[0]]['Lon'], df.iloc[route[0]]['Lat'], 'rs', markersize=12, label='START HERE')

    for i, name in enumerate(df['Vendor']):
        color = 'red' if 'Lego' in name else 'black'
        weight = 'bold' if 'Lego' in name else 'normal'
        plt.annotate(name, (df['Lon'][i], df['Lat'][i]), color=color, fontweight=weight, fontsize=9)

    plt.title("Lego-Priority Optimized Route (Bucharest)", fontsize=16)
    plt.legend()
    plt.show()

plot_priority_route(df, best_priority_route)

#cell 10, plot the convergence

def plot_priority_convergence(history_weighted):
    plt.figure(figsize=(10, 5))
    plt.plot(history_weighted, color='forestgreen', linewidth=2, label='Priority Path Distance')

    plt.title("Lego-Priority GA Convergence", fontsize=14)
    plt.xlabel("Generation")
    plt.ylabel("Physical Distance")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.annotate(f'Initial: {history_weighted[0]:.4f}', (0, history_weighted[0]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    plt.annotate(f'Final: {history_weighted[-1]:.4f}', (len(history_weighted)-1, history_weighted[-1]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, weight='bold')

    plt.legend()
    plt.show()

plot_priority_convergence(history_weighted)

#cell 11, new penalty function for reverse traversal from last one

def get_lego_last_fitness(route):
    dist = get_route_distance(route)

    lego_penalty = 0
    total_positions = len(route)

    for position, vendor_idx in enumerate(route):
        if vendor_idx in lego_indices:
            delay_score = (total_positions - position)
            lego_penalty += (delay_score**2) * 0.005

    return 1 / (dist + lego_penalty)

#cell 12, traversal

pop_size = 150
generations = 500
stagnation_limit = 50
mutation_rate = 0.1

best_dist_overall_last = float('inf')
gens_without_improvement_last = 0
history_lego_last = []

population_last = []
for _ in range(pop_size):
    remaining = [i for i in range(num_vendors) if i != START_INDEX]
    random.shuffle(remaining)
    population_last.append([START_INDEX] + remaining)

print(f"Starting 'Lego-Last' Evolution (Start: {df.iloc[START_INDEX]['Vendor']})")
print(f"{'Gen':<5} | {'Best Individual (Route)':<60} | {'Distance':<10}")
print("-" * 95)

for gen in range(generations):
    population_last = sorted(population_last, key=lambda x: get_lego_last_fitness(x), reverse=True)

    best_current_route = population_last[0]
    current_dist = get_route_distance(best_current_route)
    history_lego_last.append(current_dist)

    if current_dist < best_dist_overall_last:
        best_dist_overall_last = current_dist
        gens_without_improvement_last = 0
    else:
        gens_without_improvement_last += 1

    if gen % 10 == 0:
        print(f"{gen:<5} | {str(best_current_route):<60} | {current_dist:.4f}")

    if gens_without_improvement_last >= stagnation_limit:
        print("-" * 95)
        print(f"EARLY STOPPING: No improvement for {stagnation_limit} generations.")
        print(f"Final Convergence reached at Generation {gen}.")
        break

    new_gen = [population_last[0], population_last[1]]

    while len(new_gen) < pop_size:
        p1, p2 = random.sample(population_last[:pop_size//2], 2)
        child = crossover(p1, p2)

        if child[0] != START_INDEX:
            s_pos = child.index(START_INDEX)
            child[0], child[s_pos] = child[s_pos], child[0]

        child = constrained_mutate(child, rate=mutation_rate)
        new_gen.append(child)

    population_last = new_gen

best_lego_last_route = population_last[0]
print("-" * 95)
print(f"Final Lego-Last Route: {best_lego_last_route}")

#cell 13, plot route

def plot_lego_last_route(df, route):
    ordered_df = df.iloc[route + [route[0]]]
    plt.figure(figsize=(15, 10))

    plt.plot(ordered_df['Lon'], ordered_df['Lat'], 'o-', mfc='violet', linewidth=2, color='purple')
    plt.plot(df.iloc[route[0]]['Lon'], df.iloc[route[0]]['Lat'], 'rs', markersize=12, label='START')

    for i, name in enumerate(df['Vendor']):
        color = 'red' if 'Lego' in name else 'black'
        weight = 'bold' if 'Lego' in name else 'normal'
        plt.annotate(name, (df['Lon'][i], df['Lat'][i]), color=color, fontweight=weight, fontsize=9)

    plt.title("Lego-Last Optimized Route (Separate Vendors First)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

plot_lego_last_route(df, best_lego_last_route)

#cell 14, plot convergence

def plot_last_convergence(history_lego_last):
    plt.figure(figsize=(10, 5))
    plt.plot(history_lego_last, color='purple', linewidth=2, label='Lego-Last Path Distance')

    plt.title("Lego-Last GA Convergence History", fontsize=14)
    plt.xlabel("Generation")
    plt.ylabel("Physical Distance")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.annotate(f'Initial: {history_lego_last[0]:.4f}', (0, history_lego_last[0]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.annotate(f'Final: {history_lego_last[-1]:.4f}', (len(history_lego_last)-1, history_lego_last[-1]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, weight='bold')

    plt.legend()
    plt.show()

plot_last_convergence(history_lego_last)

#cell 16, compare the 2 oposite traversals to see which is better

def degrees_to_km(dist_deg):
    return dist_deg * 100

dist_priority_km = degrees_to_km(get_route_distance(best_priority_route))
dist_last_km = degrees_to_km(get_route_distance(best_lego_last_route))

better_route = "Lego-First" if dist_priority_km < dist_last_km else "Lego-Last"
difference = abs(dist_priority_km - dist_last_km)

print("--- FINAL LOGISTICS COMPARISON ---")
print(f"Lego-First Route Distance: {dist_priority_km:.2f} km")
print(f"Lego-Last Route Distance:  {dist_last_km:.2f} km")
print("-" * 35)
print(f"The {better_route} strategy is more efficient by {difference:.2f} km.")

plt.figure(figsize=(8, 5))
labels = ['Lego-First', 'Lego-Last']
distances = [dist_priority_km, dist_last_km]
colors = ['forestgreen', 'purple']

plt.bar(labels, distances, color=colors, alpha=0.7)
plt.ylabel('Total Distance (km)')
plt.title('Comparison: Lego-First vs. Lego-Last Strategies')

for i, v in enumerate(distances):
    plt.text(i, v + 0.5, f"{v:.2f} km", ha='center', fontweight='bold')

plt.show()
