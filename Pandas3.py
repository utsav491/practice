

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})


right = pd.DataFrame({
                     'key1': ['K0', 'K1', 'K1', 'K2'],
                     'key2': ['K0', 'K0', 'K0', 'K0'],
                     'C': ['C0', 'C1', 'C2', 'C3'],
                     'D': ['D0', 'D1', 'D2', 'D3']
                     })

print(left)
print(right)
result = pd.merge(left, right, on=['key1', 'key2'])

print(result)




result_1 = pd.merge(left, right, how='left', on=['key1', 'key2'])
print(result_1)



result_2 = pd.merge(left, right, how='right', on=['key1', 'key2'])
print(result_2)



result_3 = pd.merge(left, right, how='outer', on=['key1', 'key2'])
print(result_3)



result_4 = pd.merge(left, right, how='inner', on=['key1', 'key2'])
print(result_4)

