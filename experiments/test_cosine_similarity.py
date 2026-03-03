from tools.similarity import calculate_cosine_similarity


if __name__ == '__main__':
    # cosine_similarity = calculate_cosine_similarity(
    #     ['A boy is eating an apple.'],
    #     ['A boy is eating some meet.'],
    # )
    # print(cosine_similarity)
    # [0.6247]

    cosine_similarity = calculate_cosine_similarity(
        ['LAMP, standing for Linux, Apache, MySQL, and PHP, named in the same order.'],
        ['no'],
    )
    print(cosine_similarity)
