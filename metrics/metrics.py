import Levenshtein


def levenshtein_distance(gt_sentence, predicted_sentence):
    # gt_sentence - исходное (реальное) предложение
    # predicted_sentence - предложение, предсказанное трансформером

    # Удалите все знаки препинания из обеих последовательностей
    gt_sentence = gt_sentence.translate(str.maketrans('', '', ',.?!'))
    predicted_sentence = predicted_sentence.translate(str.maketrans('', '', ',.?!'))

    # Разделите предложения gt_sentence и predicted_sentence на слова, используя метод split()
    gt_sentence_list = gt_sentence.split()
    predicted_sentence_list = predicted_sentence.split()

    # Если списки gt_sentence_list и predicted_sentence_list имеют разную длину, расстояние Левенштейна рассчитывается следующим образом:
    #    1. Определите минимальную min_list_length и максимальную max_list_length длины этих списков
    #    2. Рассчитайте расстояние Левенштейна по следующей формуле:
    # sum_value = sum([1.0 - (Levenshtein.distance(predicted_word, word) / max(len(word), len(predicted_word)))
    #                  for word, predicted_word in zip(gt_sentence_list[:min_list_length], predicted_sentence_list[:min_list_length])
    #                  ])/max_list_length
    if len(gt_sentence_list) != len(predicted_sentence_list):
        min_list_length = min(len(gt_sentence_list), len(predicted_sentence_list))
        max_list_length = max(len(gt_sentence_list), len(predicted_sentence_list))

        sum_value = sum([1.0 - (Levenshtein.distance(predicted_word, word) / max(len(word), len(predicted_word)))
                         for word, predicted_word in
                         zip(gt_sentence_list[:min_list_length], predicted_sentence_list[:min_list_length])
                         ]) / max_list_length
        return sum_value

        # Если списки имеют одинаковую длину, рассчитываем расстояние Левенштейна
    return 1.0 - (
                Levenshtein.distance(gt_sentence, predicted_sentence) / max(len(gt_sentence), len(predicted_sentence)))