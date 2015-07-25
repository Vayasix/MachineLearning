
def bitswap(bitrow, ideal):
    output = ""
    count = 1
    length = len(bitrow)
    i = 0
    while True:
        if i < length-1:
            current_num = bitrow[i]
            next_num = bitrow[i+1]
        elif i == length-1:
            output = output + str(count)
            break

        if next_num == current_num:
            count += 1
            i += 1
        else:
            output = output + str(count)
            count = 1 #reset
            i += 1
    print output
    return abs(len(ideal) - len(output))

if __name__ == "__main__":
    #test = '0110001111001'
    ###output of bitrow : 123421
    #test_ideal = '111123121'
    ###output of ideal bitrow: 0101001110110 <- 3times swaped
    ###right answer = 3
    test = raw_input('input  bit row >> ')
    test_ideal = raw_input('input ideal row >>')
    result = bitswap(test, test_ideal)
    print result

