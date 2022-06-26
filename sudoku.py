import random
from math import sqrt, isclose


class Sudoku:

    EASY = 0.8
    NORMAL = 0.65
    HARD = 0.45
    EXTREME = 0.2

    EMPTY_CHAR = " "

    def __init__(self, size, matrix=None):
        self.block_size = sqrt(size)
        if not isclose(self.block_size % 1, 0):
            raise ValueError("Invalid dimensions")
        self.nr_columns = size
        self.nr_rows = size
        if matrix is not None:
            self._rows = matrix
        else:
            self._rows = [[self.EMPTY_CHAR for _ in range(self.nr_columns)] for _ in range(self.nr_rows)]

    def solve(self):
        # create the sudoku with all values
        rindex = 0
        cindex = 0
        available_nr_matrix = self._get_available_nr_matrix()
        while rindex < self.nr_rows:
            available_numbers = available_nr_matrix[rindex][cindex]
            # when emcountering a pre-determined number, skip it
            if available_numbers is None:
                cindex, rindex = self._next_location(cindex, rindex)
                continue
            while True:

                if len(available_numbers) == 0:
                    while True:
                        if available_nr_matrix[rindex][cindex] is not None:
                            available_nr_matrix[rindex][cindex] = list(map(str, range(1, self.nr_columns + 1)))
                        cindex -= 1
                        if cindex < 0:
                            cindex = self.nr_columns - 1
                            rindex -= 1
                        # is a pre-determined number, move one back
                        if available_nr_matrix[rindex][cindex] is None:
                            continue
                        self._rows[rindex][cindex] = " "
                        break
                    break
                random.shuffle(available_numbers)
                choice = available_numbers.pop()
                if self._can_place((cindex, rindex), choice):
                    self._rows[rindex][cindex] = choice
                    cindex, rindex = self._next_location(cindex, rindex)
                    break

    def create_random(self, difficulty: float):
        self.solve()
        # remove numbers at random depending on difficulty
        total_cells = self.nr_rows * self.nr_columns
        nr_to_remove = total_cells - int(total_cells * difficulty)
        covered_locations = set()
        for _ in range(nr_to_remove):
            location = random.randint(0, self.nr_columns - 1), random.randint(0, self.nr_columns - 1)
            while location in covered_locations:
                location = random.randint(0, self.nr_columns - 1), random.randint(0, self.nr_columns - 1)
            covered_locations.add(location)
            self._rows[location[1]][location[0]] = self.EMPTY_CHAR

    def _get_available_nr_matrix(self):
        matrix = []
        for row in self._rows:
            new_row = []
            for value in row:
                if value != self.EMPTY_CHAR:
                    # signify to not change this
                    new_row.append(None)
                else:
                    new_row.append(list(map(str, range(1, self.nr_columns + 1))))
            matrix.append(new_row)
        return matrix

    def _next_location(self, cindex, rindex):
        cindex += 1
        if cindex >= self.nr_columns:
            cindex = 0
            rindex += 1
        return cindex, rindex

    def _can_place(self, location, nr):
        # same number in row
        for value in self._rows[location[1]]:
            if value == nr:
                return False

        # same nr in column
        for row in self._rows:
            if row[location[0]] == nr:
                return False

        # same number in quadrant
        three_division1 = location[1] // 3 * 3
        for row in self._rows[three_division1: three_division1 + 3]:
            three_division2 = location[0] // 3 * 3
            for value in row[three_division2: three_division2 + 3]:
                if value == nr:
                    return False
        return True

    @classmethod
    def from_file(cls, file):
        matrix = []
        with open(file) as f:
            for rindex, line in enumerate(f):
                row = []
                for cindex, char in enumerate(line.strip()):
                    if char == "-" or char == " ":
                        row.append(cls.EMPTY_CHAR)
                    else:
                        row.append(char)
                matrix.append(row)
        sudoku = Sudoku(len(matrix[0]), matrix=matrix)
        return sudoku

    @classmethod
    def from_image(cls, file, size):
        import digits_from_image
        matrix = digits_from_image.sudoku_string_from_image(file, size)
        matrix = matrix.tolist()
        for rindex, row in enumerate(matrix):
            for cindex, value in enumerate(row):
                if value == 0:
                    matrix[rindex][cindex] = cls.EMPTY_CHAR
                else:
                    matrix[rindex][cindex] = str(value)
        sudoku = Sudoku(len(matrix[0]), matrix=matrix)
        return sudoku

    def to_file(self, file):
        with open(file, "w") as f:
            for row in self._rows:
                clear_row = "".join(map(str, row)).replace(" ", "-")
                f.write(clear_row + "\n")

    def __str__(self):
        # just for visual
        longest_column_values = []
        for index in range(self.nr_columns):
            longest_column_values.append(max([len(str(row[index])) for row in self._rows]))
        final_str_list = []
        for rindex, row in enumerate(self._rows):
            if rindex != 0 and rindex % self.block_size == 0:
                final_str_list.append("-" * len(final_str_list[-1]))
            formatted_row_values = []
            for cindex, value in enumerate(row):
                if cindex != 0 and cindex % self.block_size == 0:
                    formatted_row_values.append("|")
                formatted_row_values.append(f"{value: <{longest_column_values[cindex]}}")
            final_str_list.append(" ".join(formatted_row_values))
        return '\n'.join(final_str_list)


def generate_random(dim, difficulty, seed=None):
    if seed is not None:
        random.seed(seed)
    sudoku = Sudoku(dim)
    sudoku.create_random(difficulty)
    print(sudoku)
    exit()


def solve_image(file, size):
    import digits_from_image
    import cv2

    s = Sudoku.from_image(file, size)
    s.solve()
    original = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    original = cv2.resize(original, (500, 500))

    processed = digits_from_image.pre_process_image(original)
    colored = cv2.imread(file, cv2.IMREAD_COLOR)
    colored = cv2.resize(colored, (450, 450))
    corners = digits_from_image.find_corners_of_largest_polygon(processed)
    cropped_colored = digits_from_image.crop_and_warp(colored, corners)

    # add the numbers
    for row in s._rows:
        for value in row:
            cv2.putText(cropped_colored, value, (0, 0))
            cv2.showImage()


if __name__ == '__main__':
    solve_image("20220625_221322.jpg", 9)

