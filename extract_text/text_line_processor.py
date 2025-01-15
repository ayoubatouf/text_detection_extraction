import numpy as np


class TextLineProcessor:
    @staticmethod
    def process(result):
        if not result or not result[0]:
            return []
        return [(np.array(line[0], dtype=np.int32), line[1][0]) for line in result[0]]

    @staticmethod
    def sort(text_lines):
        return sorted(text_lines, key=lambda x: x[0][0][1])
