import collections

Node = collections.namedtuple('Node', ['left', 'right', 'value'])

# def contains(root, value):
#     if not root:
#         return False
#     if root.value == value:
#         return True
#     return contains(root.left, value) or contains(root.right, value)

# n1 = Node(value = 1, left = None, right = None)
# n3 = Node(value = 3, left = None, right = None)
# n2 = Node(value = 2, left = n1, right = n3)

# print(contains(n2, 3), n2.value)

# def find_two_sum(numbers, target_sum):
#     res = []
#     ans = [10 - i for i in numbers]
#     for i in range(len(numbers)):
#         if ans[i] in numbers[i + 1:]:
#             res.append((i, numbers.index(ans[i])))
#     return res


# print(find_two_sum([3,1,5,7,5,9], 10))

# import collections

# res = []
# for i in target_num:
#     if i not in res:
#         res.append(i)
#     else:
#         res.remove(i)

map_matrix = [
    [True, False,False],
    [True, True, False],
    [False, True, True]
]

def route_exists(from_row, from_column, to_row, to_column, map_matrix):
    fro
    if from_column == to_column and from_row == to_row:
        return True
    if 0 <= from_column < len(map_matrix[0]) and 0 <= from_row < len(map_matrix):
        if map_matrix[from_row][from_column] == True:
            return route_exists(from_row + 1, from_column, to_row, to_column, map_matrix) or route_exists(from_row, from_column + 1, to_row, to_column, map_matrix)
        else:
            return False
    return False

print(route_exists(0,1,2,1,map_matrix))