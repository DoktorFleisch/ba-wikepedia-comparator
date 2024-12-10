import networkx as nx


def get_links(page):
    links = []
    for link in page.links.values():
        links.append(link.title)
    return links


def get_backlinks(page):
    backlinks = []
    for link in page.backlinks.values():
        backlinks.append(link.title)
    return backlinks


def get_page(link, wiki):
    return wiki.page(link)


def make_graph_optimized(page, wiki):
    graph = nx.DiGraph()
    page_links = get_links(page)

    for link in page_links:
        graph.add_edge(page.title, link)

        new_page = get_page(link, wiki)
        new_page_links = get_links(new_page)
        for link in new_page_links:
            graph.add_edge(new_page.title, link)
    return graph


def pagerank(graph):
    ranked_pages = nx.pagerank(graph)
    top_n_pages = sorted(ranked_pages, key=lambda x: ranked_pages[x], reverse=True)[:5]
    return top_n_pages
