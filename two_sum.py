def new_two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

if __name__ == '__main__':
    nums = [2, 7, 11, 15, 3, 6, 8, 10, 1, 4, 5]
    target = 9
    result = new_two_sum(nums, target)
    print(f'Indices: {result}')

# Additional tests
assert new_two_sum([2, 7, 11, 15], 9) == [0, 1]
assert new_two_sum([3, 2, 4], 6) == [1, 2]
assert new_two_sum([3, 3], 6) == [0, 1]
assert new_two_sum([1, 2, 3, 4, 5], 9) == [3, 4]
assert new_two_sum([0, -1, 2, -3, 1], -2) == [3, 4]
assert new_two_sum([1, 5, 7, -1], 6) == [0, 1]
assert new_two_sum([-2, -1, 1, 2], 0) == [0, 3]
assert new_two_sum([0, 4, 3, 0], 0) == [0, 3]
assert new_two_sum([0, 4, 3, 0], 7) == [1, 2]
assert new_two_sum([5, 5, 5, 5], 10) == [0, 1]
