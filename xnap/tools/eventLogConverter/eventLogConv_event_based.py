import pandas as pd
import csv
import config 
import re
from random import shuffle 


MISSING_VALUE = -1  # placeholder for missing value (needs to be number, e.g. int or float)
MAX_EVENTS = 0      # maximum number of events for in a trace

# patterns to match date and time in timestamp
DATE_PATTERN = re.compile(r'\d{4}[-]\d{2}[-]\d{2}') 
TIME_PATTERN = re.compile(r'\d{2}[:]\d{2}[:]\d{2}')

counter_trace_instances = 0
old_status = 0

# dicts for data type lookup
data_type_dict_trace = {} 
data_type_dict_event = {} 



# converts a log file (.xes) to a .csv-file and saves it at a given outputPath ("path/filename")
def convert_XES_log_to_CSV(file, outputPath):
    
    # parse file to list of lines
    f = open(file)
    print("parsing file...")
    lines = f.readlines()
    
    if any('\t' in line for line in lines):
        lines = [line.strip('\t') for line in lines]
    if any('\n' in line for line in lines):
        lines = [line.strip('\n') for line in lines]
        
    # if file has no delimiter between tags
    if (len(lines) == 1):
        lines = []
        with open(file, newline='') as f:
            reader = csv.reader(f, delimiter='>')  
            for line in reader:   
                for list_elem in line:
                    list_elem = list_elem + ">"
                    lines.append(list_elem)

    
    # get list of trace and event attributes: [[trace_attr][event_attr]]
    print("get trace and event attributes...")
    log_attr = get_list_of_trace_and_event_attr(lines)

    trace_attr = log_attr[0]
    event_attr = log_attr[1]

    # build and fill log_list
    print("building empty log_list...")
    empty_log_list = build_log_list(lines, trace_attr, event_attr)
    print("filling log_list...")
    filled_log_list = fill_log_list(lines, trace_attr, event_attr, empty_log_list)
    
    # (optional) shuffle traces
    if (config.activate_shuffle == 1):
        print("shuffle...")
        shuffle(filled_log_list)
    
    # build dataframe from log_list
    print("building dataframe...")
    df = build_dataframe_from_log_list(filled_log_list, trace_attr, event_attr)

    # (optional) transform data (min-max-normalization, integer-mapping, timestamp)
    if (config.activate_transformation == 1):
        print("transform...")
        df = transform(df, trace_attr, event_attr)
        
    # export dataframe to .csv
    print("exporting to csv...")
    df.to_csv(outputPath + ".csv", index=False)
       
    return



def transform(df, trace_attr, event_attr):
    cols = df.columns
    
    status_cols = 0
    old_status_cols = 0
    
    
    # iterate over all columns
    for i in range(0, len(cols)): 
        col_name = cols[i].rsplit("_",1)[0]
        
        if (col_name != 'Trace' and col_name != 'Event'):
            col_suffix = cols[i].rsplit("_",1)[1]
        
        # create list of column
        col_list = df.iloc[:,i].tolist()
        

        # look up data type of column
        if (col_name == 'Trace' or col_name == 'Event'):
            data_type_col = 'int'
            
        elif (col_suffix == 'Trace'):    
            data_type_col = data_type_dict_trace.get(col_name) 
            
        elif (col_suffix == 'Event'):  
            data_type_col = data_type_dict_event.get(col_name)
            
            
        # transform column based on data type
        if (data_type_col == 'int' or data_type_col == 'float'):
            col_list = normalize_min_max(col_list)
            
        elif (data_type_col == 'string' or data_type_col == 'boolean' or data_type_col == 'id'):
            col_list = map_str_to_int(col_list)
            
        elif (data_type_col == 'date'):
            col_list = format_timestamp(col_list) 
            
        df.iloc[:,i] = col_list
        
        
        # print progress of transformation
        status_cols = get_progress_status_for_transformation(i+1, len(cols))
        if (status_cols > old_status_cols):
            print("{}% of transformation finished".format(status_cols), end="\n")
            old_status_cols = status_cols   
            
    return df



# maps each unique string to the same integer
def map_str_to_int(col_list):
    unique_vals = list(set(col_list))             
        
    key_val_tuple = zip(unique_vals, range((len(unique_vals))))
    tuple_list = list(key_val_tuple)

    if (str(MISSING_VALUE) in unique_vals):
        tup_idx = [tup[0] for tup in tuple_list].index(str(MISSING_VALUE))
        # map MISSING_VALUE to MISSING_VALUE
        tuple_list[tup_idx] = (str(MISSING_VALUE), MISSING_VALUE)

    if (MISSING_VALUE in unique_vals):
        tup_idx = [tup[0] for tup in tuple_list].index(MISSING_VALUE)
        # map MISSING_VALUE to MISSING_VALUE
        tuple_list[tup_idx] = (MISSING_VALUE, MISSING_VALUE)
        
    str_map = dict(tuple(tuple_list))
    int_col_list = [str_map.get(item,item) for item in col_list]
            
    return int_col_list
    


# scales numbers between 0 and 1 
def normalize_min_max(col_list):
    numbers = list(set(col_list))
    
    if (str(MISSING_VALUE) in numbers):
        numbers.remove(str(MISSING_VALUE))
        
    if (MISSING_VALUE in numbers):
        numbers.remove(MISSING_VALUE)
    
    for i in range(len(numbers)):
        numbers[i] = float(numbers[i])
    
    min_val = min(numbers)
    max_val = max(numbers)

    for i in range(len(col_list)):
        
        if (col_list[i] == str(MISSING_VALUE) or col_list[i] == MISSING_VALUE): 
            # map MISSING_VALUE to MISSING_VALUE
            col_list[i] = MISSING_VALUE
        else:
            old_val = float(col_list[i])
            
            if ((max_val - min_val) > 0):
                col_list[i] = (old_val - min_val)/(max_val - min_val)
            else:
                col_list[i] = min_val
                
        i += 1
        
    return col_list



# formats timestamps to 'TT.MM.YYYY HH:MM:SS'
def format_timestamp(col_list):
    for i in range(len(col_list)):
        
        if (col_list[i] == str(MISSING_VALUE) or col_list[i] == MISSING_VALUE):
            # map MISSING_VALUE to MISSING_VALUE
            col_list[i] = str(MISSING_VALUE)
        else:
            date = re.search(DATE_PATTERN, col_list[i]).group(0)
            time = re.search(TIME_PATTERN, col_list[i]).group(0)
            
            year = date.split("-")[0]
            month = date.split("-")[1]
            day = date.split("-")[2]
            
            col_list[i] = day + '.' + month + '.' + year + ' ' + time
            
    return col_list



# writes a log_list into a csv-file
def build_dataframe_from_log_list(log_list, trace_attr, event_attr):  
    count_trace_attr = len(trace_attr)
    trace_number = 0
    
    show_event_count = 0
    
    status = 0
    global old_status
    
    # list of lists where each entry represents single row in aspired dataframe 
    df_list = []  

    # build df_list from log_list
    for i in range(0, len(log_list)):
        trace_number += 1
        event_number = 0
        
        status = get_progress_status(trace_number, 2)
        if (status > old_status):
            print("{}% of processing finished".format(status), end="\n")
            old_status = status
        
        trace_attr_values = log_list[i][0:count_trace_attr]
        
        # list index of first event in trace
        event_idx = count_trace_attr
        # list index of last event in trace
        trace_end_idx = len(log_list[i])-1       

        while event_idx <= trace_end_idx:   # loop for each event in trace
            event_number += 1
            event_attr_values = log_list[i][event_idx]
            
            row_list = []
            row_list.append(trace_number)
            row_list.append(event_number)
            row_list.extend(trace_attr_values)
            row_list.extend(event_attr_values)

            df_list.append(row_list)
        
            event_idx += 1
            show_event_count += 1
    
    print("total amount of events: ", show_event_count)
    
    # build dataframe from df_list  
#    cols = ['Trace', 'Event'] + trace_attr + event_attr
    trace_attr_suffix = [a + "_Trace" for a in trace_attr]
    event_attr_suffix = [a + "_Event" for a in event_attr]
    cols = ['Trace', 'Event'] + trace_attr_suffix + event_attr_suffix
    df = pd.DataFrame.from_records(df_list, columns = cols, index = list(range(0,len(df_list))))
    
    
    # rearrange columns to have 'Trace', 'Event' and 'time:timestamp' at first position
    cols = df.columns.tolist()
    
    timestamp_col_name = [i for i in cols if i.startswith('time:timestamp')][0]
    timestamp_idx = cols.index(timestamp_col_name) 
    
    other_cols = cols[:]
    other_cols.remove('Trace')
    other_cols.remove('Event')
    other_cols.remove(timestamp_col_name)
    
    new_order = [0, 1, timestamp_idx]
    new_order += [cols.index(c) for c in other_cols]
    cols = [cols[i] for i in new_order]
    
    df = df[cols] 

    return df



# fills log_list with attribute values for all traces and events
def fill_log_list(lines, trace_attr, event_attr, log_list):
    
    trace_idx = 0
    count_trace_attr = len(trace_attr)
    event_idx = count_trace_attr
    
    in_trace = 0    # locate whether inside <trace>
    in_event = 0    # locate whether inside <event> 
    
    status = 0
    global old_status
    
    for line in lines:             
        if line.startswith('<trace'):
            in_trace = 1
            
            
        elif line.startswith('<event'):
            in_event = 1


        elif (line.startswith('<string') or line.startswith('<int') or 
              line.startswith('<float') or line.startswith('<boolean') or 
              line.startswith('<date') or line.startswith('<id')):
            if (in_trace == 1 and in_event == 0):       # fill trace attribute
                log_list = parse_attr(line, trace_idx, None, log_list, trace_attr, event_attr)
            
            elif (in_trace == 1 and in_event == 1):     # fill event attribute
                log_list = parse_attr(line, trace_idx, event_idx, log_list, trace_attr, event_attr)            
            
            
        elif line.startswith('</event'):
            event_idx += 1
            in_event = 0
                
            
        elif line.startswith('</trace'):
            trace_idx += 1
            event_idx = count_trace_attr
            in_trace = 0
            
            status = get_progress_status(trace_idx, 1)
            if (status > old_status):
                print("{}% of processing finished".format(status), end="\n")
                old_status = status
                
                
        elif line.startswith('</log'):
            break 
    
    
    return log_list



# fills log_list with single attribute value (for given attribute key)
def parse_attr(line, trace_idx, event_idx, log_list, trace_attr, event_attr): 
    
    line_tokens = line.split("key=")
    key_value_tok = line_tokens[1].split('"')
    
    attr_key = key_value_tok[1]
    attr_value = key_value_tok[3]
    
    if event_idx is None:           # fill trace attribute
        attr_idx = trace_attr.index(attr_key)
        log_list[trace_idx][attr_idx] = attr_value
         
    elif event_idx is not None:     # fill event attribute
        attr_idx = event_attr.index(attr_key)        
        log_list[trace_idx][event_idx][attr_idx] = attr_value
        
    return log_list



# builds a list structure for traces and events filled with (only) attribute keys
def build_log_list(lines, trace_attr, event_attr):
    
    log_list = None    # structure: [[trace_list],[trace_list],...]
    trace_list = None  # structure: [trace_attr_value, trace_attr_value..., [event_list], [event_list],...]
    event_list = None  # structure: [event_attr_value, event_attr_value,...]
    
    trace_attr_empty_entries = len(trace_attr) * [MISSING_VALUE]
    event_attr_empty_entries = len(event_attr) * [MISSING_VALUE]
    
    trace_number = 0
    status = 0
    global old_status
    
    for line in lines:
        if line.startswith('<log'):
            log_list = []
        
        elif line.startswith('<trace'):
            trace_list = []
            trace_list.extend(trace_attr_empty_entries)
            trace_number += 1
            
            status = get_progress_status(trace_number, 0)
            if (status > old_status):
                print("{}% of processing finished".format(status), end="\n")
                old_status = status
            
            
        elif line.startswith('<event'):
            event_list = []
            event_list.extend(event_attr_empty_entries)            
            
            
        elif line.startswith('</event'):  
            trace_list.append(event_list)  
            event_list = None
                
            
        elif line.startswith('</trace'):
            log_list.append(trace_list)
            trace_list = None
                
            
        elif line.startswith('</log'):
            break 


    return log_list



# returns a list of trace and event attributes, that has structure [[trace_attr][event_attr]]
def get_list_of_trace_and_event_attr(lines):
    
    trace_attr_list = []
    event_attr_list = []
    in_trace = 0
    in_event = 0
    
    global counter_trace_instances
    global MAX_EVENTS
    current_max_events = 0
    
    global data_type_dict_trace
    global data_type_dict_event
    
    for line in lines:
        
        if line.startswith('<trace'):
            in_trace = 1
            counter_trace_instances += 1
            
            
        elif line.startswith('<event'):
            in_event = 1
            current_max_events += 1
            
            
        elif (line.startswith('<string') or line.startswith('<int') or 
              line.startswith('<float') or line.startswith('<boolean') or 
              line.startswith('<date') or line.startswith('<id')):
            
            if (in_trace == 1):
                if (len(line.split('key')) == 1):
                    # if there is no key in line, skip this line
                    continue
                else:                    
                    attr_key = line.split('key=')[1].split('"')[1]
                                    
                data_type = line.split(" key=")[0].split("<")[1]
            
                if (in_event == 0):   # trace attribute
                    if attr_key not in trace_attr_list:
                        trace_attr_list.append(attr_key)
                        data_type_dict_trace[attr_key] = data_type
                        
                if (in_event == 1):   # event attribute
                    if attr_key not in event_attr_list:
                        event_attr_list.append(attr_key)
                        data_type_dict_event[attr_key] = data_type
                        
            else:
                # if this is a log attribute (not trace attr or event attr), skip line
                continue
                    
        elif line.startswith('</trace'):
            in_trace = 0
            if (current_max_events > MAX_EVENTS):
                MAX_EVENTS = current_max_events
            current_max_events = 0
            
        elif line.startswith('</event'):
            in_event = 0
      
    attr_list = []
    attr_list.append(trace_attr_list)
    attr_list.append(event_attr_list)     
            
    return attr_list



def get_progress_status(trace_number, function_number):  
    trace_iteration = trace_number + function_number*counter_trace_instances
    iteration_percentage = 100*( trace_iteration / (3*counter_trace_instances) )
        
    return round(iteration_percentage,0)



def get_progress_status_for_transformation(col_number, counter_columns):  
    iteration_percentage = 100*( col_number / counter_columns )
        
    return round(iteration_percentage,0)