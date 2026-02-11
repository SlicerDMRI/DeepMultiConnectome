"""Label encoding utilities for ROI pairs."""


def generate_label_dict(num_labels=85, method='symmetric'):
    """Generate encoding dictionary for ROI pair labels."""
    label_dict = {}
    if method == 'default':
        for i in range(num_labels):
            for j in range(num_labels):
                label_dict[(i, j)] = i * num_labels + j
    elif method == 'symmetric':
        index = 0
        for i in range(num_labels):
            for j in range(i, num_labels):
                label_dict[(i, j)] = index
                index += 1
    return label_dict


def generate_reverse_label_dict(num_labels=85, method='symmetric'):
    """Generate reverse dictionary for decoding ROI pair labels."""
    reverse_label_dict = {}
    if method == 'default':
        for i in range(num_labels):
            for j in range(num_labels):
                combined_label = i * num_labels + j
                reverse_label_dict[combined_label] = (i, j)
    elif method == 'symmetric':
        index = 0
        for i in range(num_labels):
            for j in range(i, num_labels):
                reverse_label_dict[index] = (i, j)
                index += 1
    return reverse_label_dict


def encode_labels(a, b, num_labels=85, method='symmetric', label_dict=None):
    """Encode ROI pair (a, b) to a single integer label."""
    if label_dict is None:
        label_dict = generate_label_dict(num_labels, method)
    if method == 'symmetric' and a > b:
        a, b = b, a
    return label_dict.get((a, b))


def decode_label(combined_label, num_labels=85, method='symmetric', reverse_label_dict=None):
    """Decode integer label to ROI pair."""
    if reverse_label_dict is None:
        reverse_label_dict = generate_reverse_label_dict(num_labels, method)
    return reverse_label_dict.get(combined_label)


def encode_labels_file(input_file, output_file, encoding_type='symmetric', num_labels=85):
    """Encode ROI pair labels from a text file."""
    label_dict = generate_label_dict(num_labels, method=encoding_type)
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                a, b = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            combined_label = encode_labels(a, b, num_labels=num_labels,
                                           method=encoding_type, label_dict=label_dict)
            outfile.write(f"{combined_label}\n")


def decode_labels_file(input_file, output_file, encoding_type='symmetric', num_labels=85):
    """Decode integer labels to ROI pairs from a text file."""
    reverse_label_dict = generate_reverse_label_dict(num_labels, method=encoding_type)
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                combined_label = int(line.strip())
            except ValueError:
                continue
            a, b = decode_label(combined_label, num_labels=num_labels,
                                method=encoding_type, reverse_label_dict=reverse_label_dict)
            outfile.write(f"{a} {b}\n")


def encode_labels_txt(input_file, output_file, encoding_type='symmetric', num_labels=85):
    """Compatibility alias for encode_labels_file."""
    encode_labels_file(input_file, output_file, encoding_type=encoding_type, num_labels=num_labels)


def convert_labels_list(labels_list, encoding_type='symmetric', mode='encode', num_labels=85):
    """Convert a list of label pairs to integers or reverse."""
    converted_list = []
    label_dict = generate_label_dict(num_labels, method=encoding_type) if mode == 'encode' else None
    reverse_label_dict = generate_reverse_label_dict(num_labels, method=encoding_type) if mode == 'decode' else None

    for labels in labels_list:
        if mode == 'encode':
            a, b = labels
            converted_label = encode_labels(a, b, num_labels=num_labels, method=encoding_type, label_dict=label_dict)
            converted_list.append(converted_label)
        elif mode == 'decode':
            combined_label = labels[0] if isinstance(labels, list) else int(labels)
            a, b = decode_label(combined_label, num_labels=num_labels, method=encoding_type, reverse_label_dict=reverse_label_dict)
            converted_list.append((a, b))
        else:
            raise ValueError("Invalid mode. Choose 'encode' or 'decode'.")

    return converted_list
