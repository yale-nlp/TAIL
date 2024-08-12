def find_matching_entries(test_depth_list, test_doc_length, json_data):
    matching_entries = []
    for depth in test_depth_list:
        for length in test_doc_length:
            # 在JSON数据中查找与当前depth和length匹配的条目
            matching_entry = next((item for item in json_data if item['depth'] == depth and item['token_length'] == length), None)
            matching_entries.append(matching_entry)
            # 如果未找到匹配条目，则抛出错误
            if matching_entry is None:
                raise ValueError(f"No matching entry found for depth {depth} and token_length {length}.")
            
            print(f"Found matching entry for depth {depth} and token_length {length}: {matching_entry}")
    return matching_entries
# 示例使用
test_depth_list = [34, 42]
test_doc_length = [100, 150, 200]
json_data = [
    {"depth": 34, "token_length": 100, "other_data": "example1"},
        {"depth": 34, "token_length": 150, "other_data": "example1"},
            {"depth": 34, "token_length": 200, "other_data": "example1"},
    {"depth": 42, "token_length": 150, "other_data": "example2"},
    {"depth": 42, "token_length": 200, "other_data": "example3"},
     {"depth": 42, "token_length": 100, "other_data": "example3"},
    # 你可以在这里添加更多数据
]

print(find_matching_entries(test_depth_list, test_doc_length, json_data))
