def unexpectedness(recommended, user_profile):

    unexpected = [
        item for item in recommended
        if item not in user_profile
    ]

    return len(unexpected) / len(recommended)