import os
import random
import re
import sys

# Probability of following a link vs. random jump
DAMPING = 0.85
# Number of random walks for sampling method
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    pages = list(corpus.keys())
    total_pages = len(pages)
    links = corpus[page]

    # If page has no outlinks, distribute probability equally to all pages
    if not links:
        return {p: 1 / total_pages for p in pages}

    # Base probability from random jumps to any page
    base_prob = (1 - damping_factor) / total_pages
    # Probability distributed among actual links
    link_prob = damping_factor / len(links)
    distribution = {p: base_prob for p in pages}

    # Add link probabilities on top of base probability
    for linked_page in links:
        distribution[linked_page] += link_prob

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    counts = {page: 0 for page in pages}

    # Start random walk from arbitrary page
    current_page = random.choice(pages)
    counts[current_page] += 1

    # Simulate random walks and track visits
    for _ in range(1, n):
        model = transition_model(corpus, current_page, damping_factor)
        weights = [model[page] for page in pages]
        current_page = random.choices(pages, weights=weights, k=1)[0]
        counts[current_page] += 1

    # Convert visit counts to probabilities
    return {page: counts[page] / n for page in pages}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    total_pages = len(pages)

    # Pages with no outlinks distribute rank equally to all pages
    links_map = {
        page: (corpus[page] if corpus[page] else set(pages))
        for page in pages
    }

    # Initialize all pages with equal rank
    ranks = {page: 1 / total_pages for page in pages}

    # Iterate until convergence
    while True:
        new_ranks = {}
        for page in pages:
            # Sum contributions from all pages linking to this page
            rank_sum = 0
            for possible_page in pages:
                if page in links_map[possible_page]:
                    rank_sum += ranks[possible_page] / len(links_map[possible_page])
            # Apply PageRank formula with damping factor
            new_ranks[page] = (1 - damping_factor) / total_pages + damping_factor * rank_sum

        # Check convergence by tracking maximum change
        max_change = max(
            abs(new_ranks[page] - ranks[page]) for page in pages
        )
        ranks = new_ranks

        if max_change < 0.001:
            break

    return ranks


if __name__ == "__main__":
    main()
