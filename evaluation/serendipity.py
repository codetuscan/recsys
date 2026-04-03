def serendipity(recommended, relevant_items, expected_items):

    serendip_items = [
        item for item in recommended
        if item in relevant_items and item not in expected_items
    ]

    return len(serendip_items) / len(recommended)