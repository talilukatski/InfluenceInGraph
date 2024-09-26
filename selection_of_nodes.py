import networkx as nx
import pandas as pd
from Praducci_simulation import create_graph, product_exposure_score


def normalize(series):
    """
    Function that normalize centrality scores
    """
    return (series - series.min()) / (series.max() - series.min())


def compute_weighted_centrality(net, weight_degree=0.4, weight_betweenness=0.4, weight_closeness=0.2):
    """
    Function to compute the weighted centrality score
    :param net: G = (V,E)
    :param weight_degree, weight_betweenness, weight_closeness: weights of the centrality indices
    :return: vector sorted by the score of combination of centrality indices
    """
    degree = pd.Series(nx.degree_centrality(net))
    betweenness = pd.Series(nx.betweenness_centrality(net))
    closeness = pd.Series(nx.closeness_centrality(net))

    degree_normalized = normalize(degree)
    betweenness_normalized = normalize(betweenness)
    closeness_normalized = normalize(closeness)

    weighted_centrality = (weight_degree * degree_normalized +
                           weight_betweenness * betweenness_normalized +
                           weight_closeness * closeness_normalized)

    return weighted_centrality.sort_values(ascending=False)


def select_influencers(net, cost_path, top_n=100, max_cost=300):
    """
    Function that select top influencers of centrality indices with cost constraint
    :param net: G = (V,E)
    :param cost_path: cost of every node
    :param top_n: how many influencers we want
    :param max_cost: how much every influencer can cost at most
    :return: influencers we would like to work on in greedy_influencer_selection
    """
    weighted_centrality = compute_weighted_centrality(net)
    costs = pd.read_csv(cost_path).set_index('user')

    affordable_influencers = [influencer for influencer in weighted_centrality.index if
                              costs.loc[influencer, 'cost'] <= max_cost]
    top_influencers = affordable_influencers[:top_n]

    return top_influencers


def compute_avg_marginal_gain(net, S, v, cost, iterations=50):
    """
    Function that compute the gain from a specific influencer v
    :param net: G = (V,E)
    :param S: group of influencers that we picked
    :param v: the influencer we want to check with the group S
    :param cost: how much the influencer v costs
    :param iterations: how many times to run mv
    :return: avg of marginal gain / cost of influencer v
    """
    total_mv = 0
    for _ in range(iterations):
        mv = product_exposure_score(net, S | {v}) - product_exposure_score(net, S)
        total_mv += mv
    avg_mv = total_mv / iterations
    return avg_mv / cost


def greedy_influencer_selection(net, cost_path, influencers, budget=1000, iterations=50):
    """
    Function for greedy algorithm to select best influencers to infect the others considering the budget
    :param net: G = (V,E)
    :param cost_path: path to the costs of all influencers we chose
    :param influencers: the best influencers we chose from the network
    :param budget: total budget to pay the influencers
    :param iterations: how many times to run mv in compute_avg_marginal_gain
    :return: best influencers to infect the others considering the budget
    """
    costs = pd.read_csv(cost_path).set_index('user')
    S = set()
    total_cost = 0

    while True:
        best_v = None
        best_value = 0

        for v in influencers:
            v_cost = costs.loc[v, 'cost']
            if total_cost + v_cost <= budget:
                avg_mv = compute_avg_marginal_gain(net, S, v, v_cost, iterations)
                if best_v is None or avg_mv > best_value:
                    best_v = v
                    best_value = avg_mv

        if best_v is None:
            break

        S.add(best_v)
        influencers.remove(best_v)
        total_cost += costs.loc[best_v, 'cost']

    return list(S)


# Example usage
NoseBook_network = create_graph('NoseBook_friendships.csv')
selected_influencers = select_influencers(NoseBook_network, 'costs.csv')
print(selected_influencers)
final_influencers = greedy_influencer_selection(NoseBook_network, 'costs.csv', selected_influencers)
print("final_influencers:", final_influencers)

