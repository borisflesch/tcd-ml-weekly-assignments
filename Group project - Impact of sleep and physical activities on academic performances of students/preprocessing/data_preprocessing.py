import pandas as pd
import numpy as np
import math

def parse_col_names(df, desired_name):
    # We drop all the columns that do not contain the desired name
    todrop = []
    for i in range(len(df.columns)):
        if desired_name not in df[i][0]:
            todrop.append(i)
    
    df = df.drop(todrop,axis=1)
    df= df.reset_index(drop=True)
    df.columns = range(df.shape[1])
    return df

def parse_bad_input(df,tolerance,cutoff_value):
    todrop = []
    
    for index,row in df.iterrows():
        if index == 0:
            continue
        unusable = 0        
        for j in range(len(df.columns)):
            try:
                if(math.isnan(int(row[j])) or int(row[j])>cutoff_value):
                    unusable += 1
            except:
                unusable += 1
                row[j]=99
        if(unusable>tolerance):
            todrop.append(index)
    return todrop

def drop_and_reset(df,todrop):
    df = df.drop(todrop)
    df= df.reset_index(drop=True)
    if(len(df.shape)>1):
        df.columns = range(df.shape[1])
    return df

def classifier_output(output):
    new_output_number = []
    new_output_matrix = []
    for i in output:
        num = float(i)
        if num< 15:
            new_output_number.append(0)
            new_output_matrix.append([1,0,0])
        else:
            if num < 20:
                new_output_number.append(1)
                new_output_matrix.append([0,1,0])
            else:
                new_output_number.append(2)
                new_output_matrix.append([0,0,1])
    return new_output_number,new_output_matrix


def parse_data(y_column=188,y_outside_range=30,type='matrix'):
    
    df=pd.read_csv("../dataset/2006SIATeensSleepRawData.csv", header=None)
    
    # We save the output column we are interested in 
    output = df.iloc[:,y_column]

    # As we are only interested in the responses to question c17, we drop everything except those
    df = parse_col_names(df,'c17')
    
    # We then check if any participant has a any skipped responses to these question
    # In case we drop them
    todrop = parse_bad_input(df,0,10)
    
    df = drop_and_reset(df,todrop)
    output = drop_and_reset(output,todrop)

    # We save the column names in diffrent arrays
    col_names = df.iloc[0,:]
    df = drop_and_reset(df,[0])

    output_col_name=output[0]
    output= drop_and_reset(output,[0])   


    # Check if there are any NAN or values outside the desired range in the output column
    todrop = []
    for i in range(len(output)):
        try:
            if math.isnan(float(output[i])) or float(output[i])>y_outside_range:
                todrop.append(i)
        except:
            todrop.append(i)

    df = drop_and_reset(df,todrop)
    output = drop_and_reset(output,todrop)

    # Create Sklearn friendly arrays
    features = np.array(df)
    
    # Tranform the outputs into classifier friendly 
    if type =='number':
        output,_ = classifier_output(output)
    if type == 'matrix':
        _,output= classifier_output(output)

    output = np.array(output)

    
    return((col_names,features),(output_col_name,output))
