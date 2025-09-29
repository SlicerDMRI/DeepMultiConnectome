def read_numbers_from_file(filename):
    """Reads numbers from a file, one per line, and returns a set of integers."""
    with open(filename, 'r') as file:
        numbers = {int(line.strip()) for line in file}
    return numbers

def write_numbers_to_file(numbers, filename):
    """Writes numbers to a file, each number on a new line."""
    with open(filename, 'w') as file:
        for number in sorted(numbers):
            file.write(f"{number}\n")

def find_unique_numbers(file1, file2, output_file):
    """Finds numbers present in file1 but not in file2 and writes them to output_file."""
    numbers_file1 = read_numbers_from_file(file1)
    numbers_file2 = read_numbers_from_file(file2)

    # Find numbers only in the first file
    unique_numbers = numbers_file1 - numbers_file2

    # Write the result to the output file
    write_numbers_to_file(unique_numbers, output_file)

# Example usage:
file1 = '/media/volume/HCP_diffusion_MV/TractCloud/tractography/subjects_tractography_output.txt'
file2 = '/media/volume/HCP_diffusion_MV/test_retest_subjects.txt'
output_file = '/media/volume/HCP_diffusion_MV/subjects_tractography_output_1000.txt'

find_unique_numbers(file1, file2, output_file)
