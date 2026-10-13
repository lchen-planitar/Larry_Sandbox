import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np

import re
import pandas as pd

def parse_lito1_srt(srt_file_path, output_csv_path):
    with open(srt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the SRT by double line breaks to isolate each frame's block
    blocks = content.strip().split('\n\n')
    telemetry_data = []
    
    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue  # Skip invalid or short blocks
            
        frame_number = lines[0]
        time_range = lines[1]
        
        # Combine everything from line 3 onwards as the text payload
        payload = " ".join(lines[2:])
        
        # Base row data with sync metrics
        row = {
            "Frame": frame_number,
            "Time_Range": time_range
        }
        
        # Regex to find DJI bracket patterns like [key : value]
        # It handles variable spacing around the colons automatically
        matches = re.findall(r'\[([^:]+)\s*:\s*([^\]]+)\]', payload)
        
        if matches:
            for key, val in matches:
                clean_key = key.strip()
                clean_val = val.strip()
                
                # Strip metric suffixes to leave raw numbers for easier graphing
                clean_val = re.sub(r'(m/s|km/h|EV|mm|m)$', '', clean_val).strip()
                
                row[clean_key] = clean_val
                
        telemetry_data.append(row)
        
    # Pandas auto-aligns the columns even if fields shift mid-flight
    df = pd.DataFrame(telemetry_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Lito 1 data extraction complete! saved {len(df)} frames to {output_csv_path}")


def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='srt file')
    parser.add_argument('output', help='csv file')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def main():
    #code
    in_file = args.input
    out_file = args.output

    parse_lito1_srt(in_file, out_file)


if __name__=="__main__":
    args = parse()
    main()