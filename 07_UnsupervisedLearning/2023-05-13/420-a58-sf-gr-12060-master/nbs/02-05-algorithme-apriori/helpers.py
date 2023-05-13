def print_itemsets(L, model):
    for Lk in L:
        print('frequent {}-itemsets：\n'.format(len(list(Lk)[0])))
        for freq_set in Lk:
            print(freq_set, 'support:', model.support_data[freq_set])

        print()
        
def print_rules(rule_list):
    for item in rule_list:
        print(item[0], "=>", item[1], "confidence: ", item[2])