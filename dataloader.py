import pandas as pd
import optparse

parse = optparse.OptionParser()
parse.add_option('-F','--input',dest = 'inputfile',action = 'store',metavar = 'time-series file',help = 'Please enter the file')
parse.add_option('-C','--col',dest = 'selectcol',action = 'store',metavar = 'col',help = 'Please enter the name of col you wanna analyse')  
parse.add_option('-A','--age',dest = 'agefilter',action = 'store',metavar = 'age filter',help = 'Please enter the max age filter')

(options,args) = parse.parse_args()

def DataLoader(file, age_column='Age(Ma)', time_filter=None, select_col=None):

    df = pd.read_excel(file)

    if time_filter:
        time_filter = float(time_filter)
        df = df[df[age_column] < time_filter]

    if select_col:
        df = df[[select_col]] 
    else:
        df = df[[age_column]]

    return df

if __name__ == '__main__':

    time_filter = float(options.agefilter) if options.agefilter else None
    
    select_cols = [options.selectcol, 'Age(Ma)']
    
    df = DataLoader(options.inputfile, 
                    time_filter=time_filter,
                    select_columns=select_cols)