'''
This Python script is designed to process a text file containing pairs of labels 
and convert them into a unique combined label. Each line of the input file 
(excluding the first line) contains two integers representing the labels, 
which range from 0 to 84. The script combines these labels using a mathematical 
encoding method to produce a single integer that uniquely represents the pair. 
The resulting combined labels are then written to a specified output text file.

Usage:
Encode using default encoding:
python label_converter.py <encode/decode> <default/symmetric> <input.txt> <output.txt>

Or within python
labels_list = [(28, 28), (75, 75), (72, 72), (30, 30), (23, 3)]
encoded_list = convert_labels_list(labels_list, encoding_type='symmetric', mode='encode')
print(encoded_list)  # Output: [1974, 3525, 3492, 2085, 269]

'''

import time
import sys

def generate_label_dict(num_labels=85, method='default'):
    """Generate a dictionary for label encoding based on the specified method."""
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

def generate_reverse_label_dict(num_labels=85, method='default'):
    """Generate a reverse dictionary for label decoding based on the specified method."""
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

def encode_labels(a, b, num_labels=85, method='default', label_dict=None):
    """Encode labels based on the specified method using a dictionary."""
    if label_dict is None:
        label_dict = generate_label_dict(num_labels, method)
    if method == 'symmetric':
        if a > b:
            a, b = b, a
    return label_dict.get((a, b))

def decode_label(combined_label, num_labels=85, method='default', reverse_label_dict=None):
    """Decode labels based on the specified method using a dictionary."""
    if reverse_label_dict is None:
        reverse_label_dict = generate_reverse_label_dict(num_labels, method)
    return reverse_label_dict.get(combined_label)

def test_encoding_decoding(num_labels=85, method='default'):
    """Test encoding and decoding of all label pairs and measure time taken."""
    # Generate the label dictionaries
    label_dict = generate_label_dict(num_labels, method)
    reverse_label_dict = generate_reverse_label_dict(num_labels, method)
    
    # Start timing
    start_time = time.time()
    
    # Test all combinations
    for a in range(num_labels):
        for b in range(num_labels):
            if method == 'symmetric' and a > b:
                continue  # Skip redundant pairs for symmetric encoding
            # Encode
            encoded_label = encode_labels(a, b, num_labels=num_labels, method=method, label_dict=label_dict)
            # Decode
            decoded_a, decoded_b = decode_label(encoded_label, num_labels=num_labels, method=method, reverse_label_dict=reverse_label_dict)
            # Check if decoding matches original
            if (a, b) != (decoded_a, decoded_b):
                print(f"Mismatch: Original ({a}, {b}) -> Decoded ({decoded_a}, {decoded_b})")
    
    # End timing
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Time taken for {method} method: {duration:.2f} seconds")

def encode_labels_txt(input_file, output_file, encoding_type, num_labels=85):
    """Process and encode labels from the input file."""
    label_dict = generate_label_dict(num_labels, method=encoding_type)
    
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    with open(output_file, 'w') as outfile:
        for line in lines[1:]:
            labels = line.strip().split()
            if len(labels) == 2:
                a = int(labels[0])
                b = int(labels[1])
                
                combined_label = encode_labels(a, b, num_labels=num_labels, method=encoding_type, label_dict=label_dict)
                
                outfile.write(f"{combined_label}\n")

def decode_labels_txt(input_file, output_file, encoding_type, num_labels=85):
    """Decode combined labels from the input file."""
    reverse_label_dict = generate_reverse_label_dict(num_labels, method=encoding_type)
    
    with open(input_file, 'r') as infile:
        combined_labels = infile.readlines()

    with open(output_file, 'w') as outfile:
        for line in combined_labels:
            combined_label = int(line.strip())
            
            a, b = decode_label(combined_label, num_labels=num_labels, method=encoding_type, reverse_label_dict=reverse_label_dict)
            
            outfile.write(f"{a} {b}\n")

def convert_labels_list(labels_list, encoding_type, mode='encode', num_labels=85):
    """Encode or decode labels from a list of label pairs."""
    converted_list = []
    label_dict = generate_label_dict(num_labels, method=encoding_type) if mode == 'encode' else None
    reverse_label_dict = generate_reverse_label_dict(num_labels, method=encoding_type) if mode == 'decode' else None
    
    for labels in labels_list:
        # if len(labels) == 2 and mode == 'encode':
        if mode == 'encode':
            a, b = labels
            converted_label = encode_labels(a, b, encoding_type, num_labels, label_dict)
            converted_list.append(converted_label)
        elif mode == 'decode':
            if type(labels)==list:
                combined_label = labels[0]
            else:
                combined_label = int(labels)
            a, b = decode_label(combined_label, encoding_type, num_labels, reverse_label_dict)
            converted_list.append((a, b))
        else:
            raise ValueError("Invalid mode. Choose 'encode' or 'decode'.")
    return converted_list

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python label_converter.py <mode> <encoding_type> <input_file> <output_file>")
        print("Mode: 'encode' to convert pairs to combined labels, 'decode' to reverse the conversion.")
        print("Encoding Type: 'default' or 'symmetric'.")
        sys.exit(1)

    mode = sys.argv[1]
    encoding_type = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    if mode == 'encode':
        encode_labels_txt(input_file, output_file, encoding_type)
    elif mode == 'decode':
        decode_labels_txt(input_file, output_file, encoding_type)
    else:
        print("Invalid mode. Please use 'encode' or 'decode'.")
        sys.exit(1)

    # # Test default encoding/decoding
    # print("\nTesting default encoding/decoding:")
    # test_encoding_decoding(num_labels=85, method='default')

    # # Test symmetric encoding/decoding
    # print("\nTesting symmetric encoding/decoding:")
    # test_encoding_decoding(num_labels=85, method='symmetric')
