import config
import csv
from random import shuffle 
import re


MISSING_VALUE = -1  # placeholder for missing value
MAX_EVENTS = 0      # maximum number of events for in a trace
MAX_EVENT_ATTR = 0  # maximum number of event attributes in an event

# TRACE_ID and EVENT_ID are attributes which will be used to identify trace and event instances (in combination with map)
TRACE_ID = "concept:name"
EVENT_ID = "concept:name"

# map ids to numeric value => index in map-list
trace_id_map = []
event_id_map = []

# dicts for data type lookup
data_type_dict_trace = {}
data_type_dict_event = {}

# patterns to match date and time in timestamp
DATE_PATTERN = re.compile(r'\d{4}[-]\d{2}[-]\d{2}')
TIME_PATTERN = re.compile(r'\d{2}[:]\d{2}[:]\d{2}')


counter_trace_instances = 0
current_event_instance = 0
old_status = 0



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
    
    del empty_log_list
    del lines
    
    # (optional) shuffle traces
    if (config.activate_shuffle == 1):
        print("shuffle...")
        shuffle(filled_log_list)
    
    # (optional) transform data (min-max-normalization, integer-mapping, timestamp)
    if (config.activate_transformation == 1):
        print("transform...")
        filled_log_list = transform(filled_log_list, trace_attr, event_attr)
    
    # write filled_log_list to outputFile
    print("writing log_list to .csv-File...")
    write_log_list_to_file(filled_log_list, trace_attr, event_attr, outputPath)

    return



# transforms data via min-max-normalization, integer-mapping and formatting of timestamps
def transform(log_list, trace_attr, event_attr):
    status_cols = 0
    old_status_cols = 0
    
    # create list of column names
    cols = []
    cols += ['Trace_ID'] + trace_attr + ['Event_ID'] + event_attr
    
    # dict to mark if attribute occurs twice (in trace_attr and event_attr)
    is_trace_and_event_attr = {}    
    
    # iterate over all columns/attributes
    c = 0
    while (c < len(cols)):
        
        col_name = cols[c]
        col_list = []
        len_trace_attr = len(trace_attr) + 1 # +1 for 'Trace_ID'
        event_idx = len_trace_attr
        
        
        # lookup column data type
#        if (col_name == 'Trace_ID' or col_name == 'Event_ID'):
#            data_type_col = 'int'
#            
        if (col_name in trace_attr or col_name == 'Trace_ID'):
            data_type_col = data_type_dict_trace.get(col_name) 
                
        elif (col_name in event_attr or col_name == 'Event_ID'):
            data_type_col = data_type_dict_event.get(col_name)
                
            
        # mark if attribute occurs twice (in trace_attr and event_attr)
        if (col_name in trace_attr and col_name in event_attr and col_name not in is_trace_and_event_attr):
            is_trace_and_event_attr[col_name] = 0
        
        
        # create column list by iterating over all traces
        t = 0
        while (t < len(log_list)):
            
            if (col_name in is_trace_and_event_attr):                   # if attribute occurs twice (in trace_attr and event_attr)
                if (is_trace_and_event_attr[col_name] == 0):
                    col_list.append(log_list[t][c])
                    
                elif (is_trace_and_event_attr[col_name] == 1):
                    event_idx = len_trace_attr
                    event_attr_idx = c - len_trace_attr
                    max_event_idx = len_trace_attr + MAX_EVENTS
                    
                    while (event_idx < max_event_idx):
                        
                        if (event_idx >= len(log_list[t])):
                            col_list.append(MISSING_VALUE)
                        else:
                            value = log_list[t][event_idx][event_attr_idx]
                            col_list.append(value)
                            
                        event_idx += 1
            
            
            elif (col_name in trace_attr or col_name == 'Trace_ID'):    # if attribute is (only) trace attribute
                col_list.append(log_list[t][c])
                
                
            elif (col_name in event_attr or col_name == 'Event_ID'):    #  if attribute is (only) event attribute                       
                event_idx = len_trace_attr
                event_attr_idx = c - len_trace_attr
                max_event_idx = len_trace_attr + MAX_EVENTS
                
                while (event_idx < max_event_idx):

                    if (event_idx >= len(log_list[t])):
                        col_list.append(MISSING_VALUE)           
                    else:
                        value = log_list[t][event_idx][event_attr_idx]
                        col_list.append(value)
                        
                    event_idx += 1
                    
            t += 1        


        # transform column based on data type
        if (data_type_col == 'int' or data_type_col == 'float'):
            col_list = normalize_min_max(col_list)

        elif (data_type_col == 'string' or data_type_col == 'boolean' or data_type_col == 'id'):
            col_list = map_str_to_int(col_list)
            
        elif (data_type_col == 'date'):
            col_list = format_timestamp(col_list) 
            

        # write transformed values back into log_list
        col_list_entry_idx = 0
        iTrace = 0

        while (iTrace < len(log_list)):  
            
            if (col_name in is_trace_and_event_attr):                   # if attribute occurs twice (in trace_attr and event_attr)
                if (is_trace_and_event_attr[col_name] == 0):
                        log_list[iTrace][c] = col_list[col_list_entry_idx]
                        col_list_entry_idx += 1
                    
                elif (is_trace_and_event_attr[col_name] == 1):
                    iEvent = len_trace_attr
                    
                    while (iEvent < max_event_idx):
                        new_val = col_list[col_list_entry_idx]
                        
                        if (iEvent >= len(log_list[iTrace]) or new_val == MISSING_VALUE or new_val == str(MISSING_VALUE)):
                            pass
                        else:
                            log_list[iTrace][iEvent][event_attr_idx] = new_val
                            
                        iEvent += 1
                        col_list_entry_idx += 1
        
        
            elif (col_name in trace_attr or col_name == 'Trace_ID'):    # if attribute is (only) trace attribute
                log_list[iTrace][c] = col_list[col_list_entry_idx]
                col_list_entry_idx += 1
         
            
            elif (col_name in event_attr or col_name == 'Event_ID'):    # if attribute is (only) event attribute
                iEvent = len_trace_attr
                
                while (iEvent < max_event_idx):
                    new_val = col_list[col_list_entry_idx]
                    
                    if (iEvent >= len(log_list[iTrace]) or new_val == MISSING_VALUE or new_val == str(MISSING_VALUE)):
                        pass
                    else:
                        log_list[iTrace][iEvent][event_attr_idx] = new_val
                        
                    iEvent += 1
                    col_list_entry_idx += 1            
            
            iTrace += 1


        if (col_name in is_trace_and_event_attr):   # if attribute occurs twice and has already been dealt with once
            is_trace_and_event_attr[col_name] += 1 
             
        # print progress of transformation
        status_cols = get_progress_status_for_transformation(c+1, len(cols))
        if (status_cols > old_status_cols):
            print("{}% of transformation finished".format(status_cols), end="\n")
            old_status_cols = status_cols   
        
        c += 1
            
    return log_list



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




# writes log_list in csv-file (row by row)
def write_log_list_to_file(log_list, trace_attr, event_attr, outputPath): 
 
    count_trace_attr = len(trace_attr) + 1
    trace_number = 0
    show_event_count = 0   
    
    status = 0
    global old_status
    
    # create list of column names
    cols = []
    cols += ['Trace_ID'] + trace_attr
    event_num = 1
    while (event_num <= MAX_EVENTS):
        cols += ['Event_ID'] + event_attr
        event_num += 1
         
    # open outputFile (specified by outputPath) and write cols
    outputFile = open(outputPath+".csv", "w", newline='')
    writer = csv.writer(outputFile)
    writer.writerow(cols)
    
    # build row_list and export it to outputFile
    for trace in log_list:
        trace_number += 1
        
        status = get_progress_status(trace_number, 2)
        if (status > old_status):
            print("{}% of processing finished".format(status), end="\n")
            old_status = status
        
        trace_attr_values = trace[0:count_trace_attr]
        
        # list index of first event in trace
        event_idx = count_trace_attr
        # list index of last event in trace
        trace_end_idx = len(trace)-1       
        
        row_list = []
        row_list.extend(trace_attr_values)

        while event_idx <= trace_end_idx:   # loop for each event in trace
            event_attr_values = trace[event_idx]
            row_list.extend(event_attr_values)
        
            event_idx += 1
            show_event_count += 1
            
        # if trace has less than MAX_EVENTS add difference
        diff = len(cols) - len(row_list)
        if (diff > 0):
            event_attr_empty_entries = (MAX_EVENT_ATTR + 1) * [MISSING_VALUE]
            row_list.extend((int)(diff/(MAX_EVENT_ATTR + 1))*event_attr_empty_entries)
        
        # write row into outputFile
        writer.writerow(row_list)
        
        del row_list

    return
    


# fills log_list with attribute values for all traces and events
def fill_log_list(lines, trace_attr, event_attr, log_list):
    
    trace_idx = 0
    count_trace_attr = len(trace_attr) + 1
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
            in_event = 0
            event_idx += 1
                
            
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
    
    attr_key = line.split('"')[1::2][0]
    attr_value = line.split('"')[1::2][1]           
    
    if event_idx is None:           # fill trace attribute
        if (attr_key == TRACE_ID):
            trace_id_numeric = trace_id_map.index(attr_value)   # trace_id_numeric is index of entry in event_id_map for given attr_value
            log_list[trace_idx][0] = trace_id_numeric
        else:
            attr_idx = trace_attr.index(attr_key)
            log_list[trace_idx][attr_idx+1] = attr_value              

    elif event_idx is not None:     # fill event attribute
        if (attr_key == EVENT_ID):
            event_id_numeric = event_id_map.index(attr_value)   
            log_list[trace_idx][event_idx][0] = event_id_numeric
        else:   
            attr_idx = event_attr.index(attr_key) 
            log_list[trace_idx][event_idx][attr_idx+1] = attr_value
            
            
    return log_list



# builds a list structure for traces and events filled with (only) attribute keys
def build_log_list(lines, trace_attr, event_attr):
    
    log_list = None    # structure: [[trace_list],[trace_list],...]
    trace_list = None  # structure: [trace_attr_value, trace_attr_value..., [event_list], [event_list],...]
    event_list = None  # structure: [event_attr_value, event_attr_value,...]
    
    trace_attr_empty_entries = (len(trace_attr) + 1)* [MISSING_VALUE]   # +1 for TRACE_ID
    event_attr_empty_entries = (MAX_EVENT_ATTR + 1) * [MISSING_VALUE]   # +1 for EVENT_ID
    
    trace_number = 0
    added_events = 0
    
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
            added_events += 1
                
            
        elif line.startswith('</trace'):
            log_list.append(trace_list)
            trace_list = None
            added_events = 0

            
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
    
    global trace_id_map
    global event_id_map
    
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
                    
                if (len(line.split('value')) == 1):
                    # if there is no value in line, skip this line
                    continue
                else:
                    attr_value = line.split('value=')[1].split('"')[1]
                                    
                data_type = line.split(" key=")[0].split("<")[1]
            
            
                if (in_event == 0):   # trace attribute
                    if (attr_key == TRACE_ID):
                        if attr_value not in trace_id_map:
                            trace_id_map.append(attr_value)
                            data_type_dict_trace['Trace_ID'] = data_type
                        continue    # attribute TRACE_ID should not be included in trace_attr_list
                    
                    if attr_key not in trace_attr_list:
                        trace_attr_list.append(attr_key)
                        data_type_dict_trace[attr_key] = data_type

                if (in_event == 1):   # event attribute
                    if (attr_key == EVENT_ID):
                        if attr_value not in event_id_map:
                            event_id_map.append(attr_value)
                            data_type_dict_event['Event_ID'] = data_type
                        continue    # attribute EVENT_ID should not be included in event_attr_list
                        
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
        
    global MAX_EVENT_ATTR
    MAX_EVENT_ATTR = len(event_attr_list)
    
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