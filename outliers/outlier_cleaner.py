#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    
    ## Obtains Tuple of Age, Net_worth, Error
    work_tup = []
    
    for index in range(0, ages.size):
      aa = ages[index]
      ee = predictions[index] - net_worths[index] 
      nn = net_worths[index]
      pp = predictions[index]
      t = [aa[0], nn[0], ee[0]]
      work_tup.append(t)
        
    work_tup = sorted(work_tup, key=lambda x: x[2])
    
    return work_tup[:81]
        
        
        
    
