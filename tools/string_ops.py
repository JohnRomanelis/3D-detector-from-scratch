def int2constString(num, string_len=6):
    '''
        Input : num (intiger)

        Output : a string with a constant length of string_len
        
            e.g  num=1, string_len=6   =>   "000001"
    '''
    
    return str(num).zfill(string_len)
    