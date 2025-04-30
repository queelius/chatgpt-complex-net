import os
from numpy.linalg import norm
import numpy as np
import networkx as nx
from datetime import datetime
import logging

def make_filename_unique(filename: str) -> str:
    """
    Make a filename unique by appending a number if the file already exists.
    """
    base, ext = filename.rsplit('.', 1)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base}_{counter}.{ext}"
        counter += 1
    return filename

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.
    """
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def jaccard_similarity(a: set, b: set) -> float:
    """
    Computes the Jaccard similarity between two sets.

    We can use this for overlapping safe urls between two documents.
    Args:
        a (set): First set.
        b (set): Second set.
    Returns:
        float: Jaccard similarity score between the two sets.
    """
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    if union == 0:
        return 0.0
    return intersection / union

def word_count_penalty(l1: int, l2: int, L0: int) -> float:
    """
    Computes a weight based on the word count.
    For a count l below L0, returns l/L0; otherwise, returns 1.
    """
    return min(1.0, l1 / L0) * min(1.0, l2 / L0)
    
def parse_date(date_str):
    """
    Parse a date string into a datetime object.
    
    Args:
        date_str (str): Date string, expected format 'YYYY-MM-DD HH:MM:SS'
        
    Returns:
        datetime.datetime: Parsed datetime object or None if parsing fails
    """
    try:
        # Try to parse the date string (format: 'YYYY-MM-DD HH:MM:SS')
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        # If parsing fails, try ISO format
        try:
            return datetime.fromisoformat(date_str)
        except (ValueError, TypeError):
            # Return None if all parsing attempts fail
            return None
    
def graph_stats(graph, top_n=10):
    """
    Compute and display detailed statistics of the NetworkX graph.

    Args:
        graph (networkx.Graph): The input graph.
        top_n (int): The number of top nodes to display for degree and centrality measures.

    Returns:
        dict: A dictionary containing various graph statistics.
    """
    stats = {}
    stats['Number of Nodes'] = graph.number_of_nodes()
    stats['Number of Edges'] = graph.number_of_edges()
    stats['Density'] = nx.density(graph)
    connected = nx.is_connected(graph.to_undirected())
    stats['Connected'] = connected
    if connected:
        stats['Average Shortest Path Length'] = nx.average_shortest_path_length(
            graph.to_undirected())
        stats['Graph Diameter'] = nx.diameter(graph.to_undirected())
    try:
        stats['Degree'] = {}
        stats['Degree']['Average'] = sum(dict(graph.degree()).values(
        )) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        stats['Degree']['Max'] = max(
            dict(graph.degree()).values()) if dict(graph.degree()) else 0
        stats['Degree']['Min'] = min(
            dict(graph.degree()).values()) if dict(graph.degree()) else 0
        top_degree = sorted(dict(graph.degree()).items(),
                            key=lambda x: x[1], reverse=True)[:top_n]
        stats['Degree'][f'Top {top_n}'] = [
            f"{url} ({degree})" for url, degree in top_degree]
    except Exception as e:
        logging.warning(f"Could not compute degree measures: {e}")

    try:
        degree_centrality = nx.degree_centrality(graph)
        stats['Degree Centrality'] = {}
        stats['Degree Centrality']['Min'] = min(
            degree_centrality.values()) if degree_centrality else 0
        stats['Degree Centrality']['Max'] = max(
            degree_centrality.values()) if degree_centrality else 0
        stats['Degree Centrality']['Average'] = sum(
            degree_centrality.values()) / len(degree_centrality) if degree_centrality else 0
        top_degree_cent = sorted(degree_centrality.items(
        ), key=lambda x: x[1], reverse=True)[:top_n]
        stats['Degree Centrality'][f'Top {top_n}'] = [
            f"{url} ({centrality:.4f})" for url, centrality in top_degree_cent]
    except Exception as e:
        logging.warning(f"Could not compute centrality measures: {e}")

    try:
        betweenness_centrality = nx.betweenness_centrality(graph)
        stats['Betweenness Centrality'] = {}
        stats['Betweenness Centrality']['Min'] = min(
            betweenness_centrality.values()) if betweenness_centrality else 0
        stats['Betweenness Centrality']['Max'] = max(
            betweenness_centrality.values()) if betweenness_centrality else 0
        stats['Betweenness Centrality']['Average'] = sum(betweenness_centrality.values(
        )) / len(betweenness_centrality) if betweenness_centrality else 0
        top_betweenness = sorted(betweenness_centrality.items(
        ), key=lambda x: x[1], reverse=True)[:top_n]
        stats['Betweenness Centrality'][f'Top {top_n}'] = [
            f"{url} ({centrality:.4f})" for url, centrality in top_betweenness]
    except Exception as e:
        logging.warning(
            f"Could not compute betweenness centrality measures: {e}")

    try:
        closeness_centrality = nx.closeness_centrality(graph)
        stats['Closeness Centrality'] = {}
        stats['Closeness Centrality']['Min'] = min(
            closeness_centrality.values()) if closeness_centrality else 0
        stats['Closeness Centrality']['Max'] = max(
            closeness_centrality.values()) if closeness_centrality else 0
        stats['Closeness Centrality']['Average'] = sum(closeness_centrality.values(
        )) / len(closeness_centrality) if closeness_centrality else 0
        top_closeness = sorted(closeness_centrality.items(
        ), key=lambda x: x[1], reverse=True)[:top_n]
        stats['Closeness Centrality'][f'Top {top_n}'] = [
            f"{url} ({centrality:.4f})" for url, centrality in top_closeness]
    except Exception as e:
        logging.warning(
            f"Could not compute closeness centrality measures: {e}")

    try:
        eigenvector_centrality = nx.eigenvector_centrality(graph)
        stats['Eigenvector Centrality'] = {}
        stats['Eigenvector Centrality']['Min'] = min(
            eigenvector_centrality.values()) if eigenvector_centrality else 0
        stats['Eigenvector Centrality']['Max'] = max(
            eigenvector_centrality.values()) if eigenvector_centrality else 0
        stats['Eigenvector Centrality']['Average'] = sum(eigenvector_centrality.values(
        )) / len(eigenvector_centrality) if eigenvector_centrality else 0
        top_eigenvector = sorted(eigenvector_centrality.items(
        ), key=lambda x: x[1], reverse=True)[:top_n]
        stats['Eigenvector Centrality'][f'Top {top_n}'] = [
            f"{url} ({centrality:.4f})" for url, centrality in top_eigenvector]
    except Exception as e:
        logging.warning(
            f"Could not compute eigenvector centrality measures: {e}")

    try:
        pagerank_centrality = nx.pagerank(graph)
        stats['PageRank Centrality'] = {}
        stats['PageRank Centrality']['Min'] = min(
            pagerank_centrality.values()) if pagerank_centrality else 0
        stats['PageRank Centrality']['Max'] = max(
            pagerank_centrality.values()) if pagerank_centrality else 0
        stats['PageRank Centrality']['Average'] = sum(pagerank_centrality.values(
        )) / len(pagerank_centrality) if pagerank_centrality else 0
        top_pagerank = sorted(pagerank_centrality.items(),
                              key=lambda x: x[1], reverse=True)[:top_n]
        stats['PageRank Centrality'][f'Top {top_n}'] = [
            f"{url} ({centrality:.4f})" for url, centrality in top_pagerank]
    except Exception as e:
        logging.warning(f"Could not compute PageRank centrality measures: {e}")

    try:
        harmonic_centrality = nx.harmonic_centrality(graph)
        stats['Harmonic Centrality'] = {}
        stats['Harmonic Centrality']['Min'] = min(
            harmonic_centrality.values()) if harmonic_centrality else 0
        stats['Harmonic Centrality']['Max'] = max(
            harmonic_centrality.values()) if harmonic_centrality else 0
        stats['Harmonic Centrality']['Average'] = sum(harmonic_centrality.values(
        )) / len(harmonic_centrality) if harmonic_centrality else 0
        top_harmonic = sorted(harmonic_centrality.items(),
                              key=lambda x: x[1], reverse=True)[:top_n]
        stats['Harmonic Centrality'][f'Top {top_n}'] = [
            f"{url} ({centrality:.4f})" for url, centrality in top_harmonic]
    except Exception as e:
        logging.warning(f"Could not compute harmonic centrality measures: {e}")

    try:
        comps = nx.connected_components(graph.to_undirected())
        comps_sizes = [len(c) for c in comps]
        stats['Connected Components'] = {}
        stats['Connected Components']['Average'] = sum(
            comps_sizes) / len(comps_sizes) if comps_sizes else 0
        stats['Connected Components']['Number'] = nx.number_connected_components(
            graph.to_undirected())
        stats['Connected Components'][f'Top {top_n} Sizes'] = sorted(
            comps_sizes, reverse=True)[:top_n]
    except Exception as e:
        logging.warning(f"Could not compute graph components measures: {e}")

    try:
        clustering_coeffs = nx.clustering(graph.to_undirected())
        stats['Clustering Coefficients'] = {}
        stats['Clustering Coefficients']['Min'] = min(
            clustering_coeffs.values()) if clustering_coeffs else 0
        stats['Clustering Coefficients']['Max'] = max(
            clustering_coeffs.values()) if clustering_coeffs else 0
        stats['Clustering Coefficients']['Average'] = sum(
            clustering_coeffs.values()) / len(clustering_coeffs) if clustering_coeffs else 0
        top_clustering = sorted(clustering_coeffs.items(
        ), key=lambda x: x[1], reverse=True)[:top_n]
        stats['Clustering Coefficients'][f'Top {top_n}'] = [
            f"{url} ({coeff:.4f})" for url, coeff in top_clustering]
    except Exception as e:
        logging.warning(
            f"Could not compute clustering coefficients measures: {e}")

    return stats

