# Alejandro Luna

import cpmpy as cp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

"""
Original Description:

The problem is to timetable a party at a yacht club. Certain boats are to be designated hosts,
and the crews of the remaining boats in turn visit the host boats for several successive halfhour
periods. The crew of a host boat remains on board to act as hosts while the crew of a
guest boat together visits several hosts. Every boat can only hold a limited number of people
at a time (its capacity) and crew sizes are different. The total number of people aboard a
boat, including the host crew and guest crews, must not exceed the capacity. A guest boat
cannot not revisit a host and guest crews cannot meet more than once.
The problem the rally organizer is facing is that of minimizing the number of host boats. Find
the visit schedule and the host designations.
"""
#PROBLEM INSTANCE
n_boats = 12 #number of boats
available_halfhours = 2 #number of available half-hour periods for visits
max_crew_size = 20 #maximum crew size per boat
capacity_max_multiplier = 5 #maximum multiplier for boat capacity beyond crew size

#Random crew sizes and capacities
np.random.seed(42) 
crew_size = np.random.randint(4, max_crew_size, n_boats)
crew_size = cp.cpm_array(crew_size)
capacity = crew_size + np.random.randint(0, capacity_max_multiplier * max_crew_size, n_boats)
capacity = cp.cpm_array(capacity)

#----------------------------------
#MODEL
boats = cp.boolvar((available_halfhours, n_boats, n_boats)) #visits between boats
model = cp.Model()

#Decision variables for host assignment
hosts = cp.boolvar(n_boats)

#----------------------------------
# CONSTRAINTS

#C1: Capacity limits & no self-visits
#Each boat has a maximum capacity, including its own crew and any visiting guests; a boat cannot visit itself
for t in range(available_halfhours):
    for b in range(n_boats):
        #the total number of people on boat b at time t must not exceed its capacity
        model += (cp.sum(boats[t, :, b] * crew_size) + crew_size[b] <= capacity[b])
        #a boat cannot be its own guest
        model += (boats[t, b, b] == 0)

#C2: Hosts must have guests, and non-hosts cannot have guests
for b in range(n_boats):
    #if boat b is a host, it must have guests visiting at some point
    model += (hosts[b].implies(cp.sum(boats[:, :, b]) > 0))
    #if boat b is not a host, no guests can visit it
    model += ((~hosts[b]).implies(cp.sum(boats[:, :, b]) == 0))

#C3: Hosts cannot visit other boats
for b in range(n_boats):
    model += (hosts[b].implies(cp.sum(boats[:, b, :]) == 0))

#C4: Guests must visit at least one host
for b in range(n_boats):
    model += ((~hosts[b]).implies(cp.sum(boats[:, b, :]) > 0))

#C5: Each guest visits a host exactly once
#if boat i is a guest and boat j is a host, then i must visit j exactly once across all time slots
for i in range(n_boats):
    for j in range(n_boats):
        if i != j:
            model += ((~hosts[i]) & hosts[j]).implies(cp.sum(boats[:, i, j]) == 1)

#C6: Each guest can visit only one boat per time slot
for ah in range(available_halfhours):
    for i in range(n_boats):
        model += (~hosts[i]).implies(cp.sum(boats[ah, i, :]) <= 1)

#C7: Guest boats should not meet another guest boat more than once
#a meeting happens if two guest boats visit the same host in the same time slot
for i in range(n_boats):
    for k in range(i + 1, n_boats):
        model += (
            (~hosts[i]) & (~hosts[k]) #both boats must be guests
        ).implies(
            cp.sum([
                boats[t, i, h] * boats[t, k, h] #they both visit host h at time t
                for t in range(available_halfhours)
                for h in range(n_boats)
            ])
            <= 1 #at most 1 (they meet at most once)
        )

#C8: Enforce at least 1 host
model += (cp.sum(hosts) > 0)

#C9: Not all boats can be hosts
model += (cp.sum(hosts) < n_boats)

#----------------------------------
#OBJECTIVE
#Maximize the number of guests (minimizing hosts)
model.maximize(cp.sum(~hosts))

#----------------------------------
#SOLVE
if model.solve():
    print("Solution found!\n")

    #VALUES
    hosts_values = [hosts[b].value() for b in range(n_boats)]
    print("Hosts (marked with True):")
    print(hosts_values)
    for t in range(available_halfhours): #visitations per half-hour
        print(f"\nHalf-hour period {t+1}:")
        for i in range(n_boats):
            for j in range(n_boats):
                if boats[t, i, j].value() == 1:
                    print(f"Boat {i+1} (guest) visits Boat {j+1} (host)")
                    
    
    #GRAPH
    graphs = []
    for t in range(available_halfhours):
        G = nx.DiGraph()
        for b in range(n_boats): #nodes for each boat
            color = "red" if hosts[b].value() else "blue"
            G.add_node(b + 1, color=color, crew_size=crew_size[b], capacity=capacity[b])
        for i in range(n_boats): #directed edges for visits
            for j in range(n_boats):
                if boats[t, i, j].value() == 1:
                    G.add_edge(i + 1, j + 1, time=t+1)
        graphs.append(G)
    fixed_pos = {i + 1: (np.cos(2 * np.pi * i / n_boats), np.sin(2 * np.pi * i / n_boats)) for i in range(n_boats)}
    for t, G in enumerate(graphs):
        fig, ax = plt.subplots(figsize=(10, 6))
        node_colors = [G.nodes[n]["color"] for n in G.nodes()]
        nx.draw(
            G, fixed_pos, with_labels=True, node_color=node_colors, edge_color="black", 
            ax=ax, font_weight='bold', node_size=1000, width=2, arrows=True, alpha=0.8
            )
        ax.set_title(f"Half-hour period {t+1}")
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label="Host")
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label="Guest")
        ax.legend(handles=[red_patch, blue_patch]) #legend for hosts and guests
        #Add crew size and capacity labels for all boats
        for node in G.nodes():
            crew_count = G.nodes[node]["crew_size"]
            capacity_count = G.nodes[node]["capacity"]
            ax.text(fixed_pos[node][0], fixed_pos[node][1] + 0.05, f"{crew_count}/{capacity_count}",
                    horizontalalignment="center", fontweight='bold', fontsize=10, color="black")
            if G.nodes[node]["color"] == "red":
                total_visitors = crew_count + sum(G.nodes[v]["crew_size"] for v in G.predecessors(node))
                ax.text(fixed_pos[node][0], fixed_pos[node][1] + 0.15, f"{total_visitors}/{capacity_count}",
                        horizontalalignment="center", fontweight='bold', fontsize=12, color="red")
        plt.tight_layout()
        plt.show()

else:
    print("No solution (UNSAT).")

