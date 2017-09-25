# ==============Group Members==================================
# Michelle Becerra
# Amirah Anwar
# Reetinder Kaur

def main():
    # ==============Process Data==================================

    # #load txt file into a ndarray
	f = open("clusters.txt", 'r')
	result_matrix = []
	for line in f.readlines():
	    values_as_strings = line.split(',')
	    result_matrix.append(map(float, values_as_strings))

	print result_matrix

if __name__ == "__main__":
    main()
