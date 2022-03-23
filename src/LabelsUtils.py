def labels_map():
    label_map = {'O': 0}
    cnt = 0
    with open('../data/example/train_500.txt', 'r', encoding='utf-8') as f:
        for item in f.readlines():
            label = item.strip().split()
            if len(label) > 0:
                label = label[-1]
            else:
                continue
            if label not in label_map:
                cnt += 1
                label_map[label] = cnt
    return label_map


def labels_unmap():
    labels_mapping = labels_map()
    return {v: k for k, v in labels_mapping.items()}


if __name__ == "__main__":
    labels_unmapping = labels_unmap()
    print(1)