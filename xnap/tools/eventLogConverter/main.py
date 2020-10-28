import eventLogConv_event_based as converter
#import eventLogConv_trace_based as converter


if __name__ == "__main__":
    # execute only if run as a script
    # python main.py input/test.xes C:\Users\Username\Desktop
    
    
    dataset = "RTFM"
    file = "./input_files/" + dataset + ".xes"
    output = "./output_files/" + dataset 
    
    converter.convert_XES_log_to_CSV(file, output)
    
