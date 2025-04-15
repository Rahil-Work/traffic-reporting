# project/reporting/excel_report.py
import pandas as pd
import re # Import regular expressions
from datetime import datetime, timedelta, time as datetime_time
from config import (
    EXCEL_REPORT_NAME, VEHICLE_ID_MAP, ALL_VEHICLE_TYPES,
    REPORTING_DIRECTIONS
)
from core.utils import debug_print, get_display_direction, format_timestamp

# --- Optimized Parsing Helper using Regex ---
def parse_detection_string(event_str):
    """
    Parses the detection string using regex to extract components robustly.
    Returns: tuple (vehicle_type, event_action, direction_to, direction_from)
             Returns None for components not found. direction_to/from are UPPERCASE.
    """
    vehicle_type = None
    event_action = None # ENTRY, EXIT, TIMEOUT, FORCED_EXIT, UNKNOWN
    direction_to = None
    direction_from = None

    # Pattern 1: Entry "VEHICLE TYPE ENTERED FROM DIRECTION"
    match_entry = re.match(r"^(.*)\s+ENTERED\s+FROM\s+(\w+)$", event_str, re.IGNORECASE)
    if match_entry:
        vehicle_type = match_entry.group(1).strip()
        event_action = "ENTRY"
        # For entry, 'direction_from' is the one mentioned, 'direction_to' is implied 'IN_PROGRESS'
        direction_from = match_entry.group(2).strip().upper()
        direction_to = "IN_PROGRESS" # Use a consistent marker
        return vehicle_type, event_action, direction_to, direction_from

    # Pattern 2: Exit/Timeout/Forced "VEHICLE TYPE {ACTION} TO {DIR_TO} [(FROM {DIR_FROM})]"
    # Covers "EXITED TO", "TIMED OUT", "FORCED EXIT"
    # Makes the "(FROM DIR)" part optional
    match_exit = re.match(r"^(.*)\s+(EXITED\s+TO|TIMED\s+OUT|FORCED\s+EXIT)\s+([^\(]+?)(\s+\(FROM\s+(\w+)\))?$", event_str, re.IGNORECASE)
    if match_exit:
        vehicle_type = match_exit.group(1).strip()
        action_keyword = match_exit.group(2).strip().upper()
        direction_to = match_exit.group(3).strip().upper() # Direction or status like TIMEOUT/FORCED
        # Group 5 captures the direction inside (FROM ...) if present
        direction_from = match_exit.group(5).strip().upper() if match_exit.group(5) else None # Use None if FROM not found

        # Determine specific event action
        if "EXITED TO" in action_keyword: event_action = "EXIT"
        elif "TIMED OUT" in action_keyword: event_action = "TIMEOUT"; direction_to = "TIMEOUT" # Standardize direction_to
        elif "FORCED EXIT" in action_keyword: event_action = "FORCED_EXIT"; direction_to = "FORCED" # Standardize direction_to
        else: event_action = "UNKNOWN" # Should not happen with this regex

        return vehicle_type, event_action, direction_to, direction_from

    # If neither pattern matched
    debug_print(f"Could not parse event string with regex: '{event_str}'")
    return None, "UNKNOWN", None, None



# project/reporting/excel_report.py
import pandas as pd
import re # Import regular expressions
from datetime import datetime, timedelta, time as datetime_time
from config import (
    EXCEL_REPORT_NAME, VEHICLE_ID_MAP, ALL_VEHICLE_TYPES,
    REPORTING_DIRECTIONS
)
from core.utils import debug_print, get_display_direction, format_timestamp

# --- Optimized Parsing Helper using Regex ---
def parse_detection_string(event_str):
    """
    Parses the detection string using regex to extract components robustly.
    Returns: tuple (vehicle_type, event_action, direction_to, direction_from)
             Returns None for components not found. direction_to/from are UPPERCASE.
    """
    vehicle_type = None; event_action = None; direction_to = None; direction_from = None

    # Pattern 1: Entry "VEHICLE TYPE ENTERED FROM DIRECTION"
    match_entry = re.match(r"^(.*)\s+ENTERED\s+FROM\s+(\w+)$", event_str, re.IGNORECASE)
    if match_entry:
        vehicle_type=match_entry.group(1).strip(); event_action="ENTRY"
        direction_from=match_entry.group(2).strip().upper(); direction_to="IN_PROGRESS"
        return vehicle_type, event_action, direction_to, direction_from

    # Pattern 2: Exit/Timeout/Forced "VEHICLE TYPE {ACTION} TO {DIR_TO} [(FROM {DIR_FROM})]"
    match_exit = re.match(r"^(.*)\s+(EXITED\s+TO|TIMED\s+OUT|FORCED\s+EXIT)\s+([^\(]+?)(\s+\(FROM\s+(\w+)\))?$", event_str, re.IGNORECASE)
    if match_exit:
        vehicle_type=match_exit.group(1).strip(); action_keyword=match_exit.group(2).strip().upper()
        direction_to=match_exit.group(3).strip().upper()
        direction_from = match_exit.group(5).strip().upper() if match_exit.group(5) else None
        if "EXITED TO" in action_keyword: event_action = "EXIT"
        elif "TIMED OUT" in action_keyword: event_action = "TIMEOUT"; direction_to = "TIMEOUT"
        elif "FORCED EXIT" in action_keyword: event_action = "FORCED_EXIT"; direction_to = "FORCED"
        else: event_action = "UNKNOWN"
        return vehicle_type, event_action, direction_to, direction_from

    debug_print(f"Could not parse event string with regex: '{event_str}'");
    return None, "UNKNOWN", None, None


# --- Main Report Function ---
def create_excel_report(detection_events, completed_paths_data, start_datetime):
    """
    Creates a comprehensive Excel report. Includes valid entry/exit pairs
    in the 'Complete Paths' sheet.
    Args:
        detection_events (list): List of dictionaries for ENTRY/EXIT/etc. logs. Used for Vehicle Logs & Directional sheets.
        completed_paths_data (list): List of dictionaries from VehicleTracker.get_completed_paths(). Used for filtered Complete Paths sheet.
        start_datetime (datetime): The starting timestamp for interval calculations.
    """
    print(f"\n--- create_excel_report ---")
    print(f"Received {len(detection_events)} detection events for logs.")
    print(f"Received {len(completed_paths_data)} completed path summaries from tracker.")
    if not detection_events and not completed_paths_data:
         print("WARNING: No detection events or completed paths received!")
         return None

    all_valid_vehicle_types = list(VEHICLE_ID_MAP.keys())
    vehicle_logs_data = []          # For the detailed Vehicle Logs sheet
    processed_movements = []        # For aggregating directional counts
    complete_paths_report_data = [] # For the FILTERED Complete Paths sheet
    vehicle_entry_info = {}         # Helper {vehicle_id: {entry_info}}

    # --- Process Events for Vehicle Logs and Directional Counts ---
    # This loop populates vehicle_logs_data and processed_movements based on the detailed events
    for event in detection_events:
        event_str = event.get('detection', ''); timestamp_str_ms = event.get('timestamp', '')
        vehicle_id = event.get('vehicle_id', 'UnknownID'); status = event.get('status', 'unknown')

        # Reconstruct full datetime
        try:
            if len(timestamp_str_ms)>6: time_part,ms_part=timestamp_str_ms[:6],timestamp_str_ms[6:]; base_time=datetime.strptime(time_part,"%H%M%S").time(); microseconds=int(ms_part.ljust(3,'0'))*1000; event_time_part=base_time.replace(microsecond=microseconds)
            else: event_time_part=datetime.strptime(timestamp_str_ms,"%H%M%S").time()
            event_timestamp = datetime.combine(start_datetime.date(), event_time_part)
        except ValueError: debug_print(f"Skip Log: Bad timestamp '{timestamp_str_ms}'"); continue

        # Parse the event string
        vehicle_type, event_action, direction_to, direction_from = parse_detection_string(event_str)

        if event_action == "UNKNOWN" or vehicle_type is None: continue # Skip unparsed

        # Determine directions for logging
        direction_to_log = direction_to
        direction_from_log = direction_from

        if event_action == "ENTRY":
            vehicle_entry_info[vehicle_id] = {'type': vehicle_type, 'entry_dir': direction_from, 'entry_time': event_timestamp}
            direction_to_log = 'IN_PROGRESS' # Standardize log display
        elif event_action in ["EXIT", "TIMEOUT", "FORCED_EXIT", "UNKNOWN_EXIT_PARSE"]:
             # Try to get 'from' direction if regex missed it (e.g., timeout/forced log format)
             if direction_from is None:
                 direction_from = vehicle_entry_info.get(vehicle_id, {}).get('entry_dir', 'UNKNOWN')
             direction_from_log = direction_from # Use the determined 'from' for the log

             # Add to processed movements ONLY if it was a valid, standard exit
             if event_action == "EXIT" and direction_to not in ['UNKNOWN', 'TIMEOUT', 'FORCED']:
                 entry_info_mov = vehicle_entry_info.get(vehicle_id)
                 # Use the FROM direction determined for this EXIT event for consistency
                 if entry_info_mov and entry_info_mov['entry_dir'] == direction_from:
                     processed_movements.append({
                         'vehicle_type': vehicle_type,
                         'direction_from': get_display_direction(direction_from), # Use determined 'from'
                         'direction_to': get_display_direction(direction_to),
                         'timestamp': event_timestamp })

        # --- Add parsed event to Vehicle Logs ---
        object_id_code = VEHICLE_ID_MAP.get(vehicle_type, '')
        ts_display = event_timestamp.strftime('%m/%d/%y %H:%M:%S.%f')[:-3]
        d_display = event_timestamp.strftime('%m/%d/%Y'); t_display = event_timestamp.strftime('%H:%M:%S.%f')[:-3]
        vehicle_logs_data.append({
            'Object ID':object_id_code, 'Object Name':vehicle_type, 'Event':event_action,
            'Direction From':get_display_direction(direction_from_log),
            'Direction To':get_display_direction(direction_to_log),
            'Timestamp':ts_display, 'Date':d_display, 'Time':t_display,
            'Vehicle Tracker ID':vehicle_id, 'Status':status
            })
    # --- End Event Loop for Logs/Movements ---


    # --- Prepare FILTERED Complete Paths Data from Tracker Summaries ---
    valid_complete_paths_count = 0
    if completed_paths_data: # Use the data passed directly from the tracker
        for path in completed_paths_data:
            entry_t = path.get('entry_time'); exit_t = path.get('exit_time')
            status = path.get('status'); entry_dir = path.get('entry_direction'); exit_dir = path.get('exit_direction')
            t_in_int = path.get('time_in_intersection')

            # --- FILTERING CONDITION ---
            is_valid_entry_dir = entry_dir and entry_dir not in ['UNKNOWN', None]
            is_valid_exit_dir = exit_dir and exit_dir not in ['UNKNOWN', 'TIMEOUT', 'FORCED', None]
            if status == 'exited' and is_valid_entry_dir and is_valid_exit_dir and entry_t and exit_t:
                valid_complete_paths_count += 1
                v_type = path.get('type', 'UnkType')
                entry_dir_disp = get_display_direction(entry_dir); exit_dir_disp = get_display_direction(exit_dir)
                entry_time_str = entry_t.strftime('%H:%M:%S.%f')[:-3]; exit_time_str = exit_t.strftime('%H:%M:%S.%f')[:-3]
                complete_path_str = f"{entry_dir_disp} TO {exit_dir_disp}"
                complete_paths_report_data.append({
                    'Vehicle Type': v_type, 'Entry Direction': entry_dir_disp, 'Exit Direction': exit_dir_disp,
                    'Entry Time': entry_time_str, 'Exit Time': exit_time_str,
                    'Time in Intersection (s)': round(t_in_int, 2) if t_in_int is not None else 'N/A',
                    'Complete Path': complete_path_str, 'Status': status })
            # --- END FILTERING ---

    total_completed = len(completed_paths_data) if completed_paths_data else 0
    debug_print(f"Filtered complete paths: Included {valid_complete_paths_count} valid paths out of {total_completed} total summaries.")
    # --- End Complete Paths Preparation ---


    # --- Create Excel File ---
    try:
        with pd.ExcelWriter(EXCEL_REPORT_NAME, engine='xlsxwriter') as writer:
            # Sheet: Vehicle Logs (Uses data parsed from detection_events)
            if vehicle_logs_data:
                 log_cols = ['Timestamp','Date','Time','Object ID','Object Name','Event','Direction From','Direction To','Status','Vehicle Tracker ID']
                 df_logs = pd.DataFrame(vehicle_logs_data)[log_cols]; df_logs.to_excel(writer, sheet_name='Vehicle Logs', index=False); debug_print(f"Created 'Vehicle Logs' sheet ({len(df_logs)} rows).")
            else: debug_print("No data for 'Vehicle Logs' sheet.")

            # Sheet: Complete Paths (Uses FILTERED data formatted from completed_paths_data)
            if complete_paths_report_data: # Use the filtered list
                path_cols = ['Vehicle Type','Entry Direction','Exit Direction','Entry Time','Exit Time','Time in Intersection (s)','Complete Path','Status']
                df_complete = pd.DataFrame(complete_paths_report_data)[path_cols]; df_complete.to_excel(writer, sheet_name='Complete Paths', index=False); debug_print(f"Created 'Complete Paths' sheet ({len(df_complete)} rows).")
            else: debug_print("No data for 'Complete Paths' sheet (after filtering).")

            # Sheet: Intervals
            intervals = []; report_dur_hrs = 7; end_limit = start_datetime + timedelta(hours=report_dur_hrs); curr_interval_start = start_datetime
            while curr_interval_start < end_limit:
                interval_end = min(curr_interval_start + timedelta(minutes=15), end_limit); intervals.append({'Date': curr_interval_start.strftime('%m/%d/%Y'), 'Time Intervals': f"{curr_interval_start.strftime('%H:%M:%S')} - {interval_end.strftime('%H:%M:%S')}", 'Start Time': curr_interval_start.strftime('%H:%M:%S'), 'End Time': interval_end.strftime('%H:%M:%S'), 'Interval Start DT': curr_interval_start, 'Interval End DT': interval_end}); curr_interval_start += timedelta(minutes=15)
            if intervals: df_intervals = pd.DataFrame(intervals); df_intervals[['Date', 'Time Intervals', 'Start Time', 'End Time']].to_excel(writer, sheet_name='Intervals', index=False); debug_print("Created 'Intervals' sheet.")
            else: debug_print("No data for 'Intervals' sheet.")

            # Sheets: Directional Counts (Uses processed_movements parsed from detection_events)
            if processed_movements and intervals:
                debug_print(f"Processing {len(processed_movements)} movements for directional sheets...")
                df_moves = pd.DataFrame(processed_movements)
                for from_dir_rep, to_dirs_rep in REPORTING_DIRECTIONS.items():
                    from_dir_match = from_dir_rep.split("From ")[-1]; sheet_data = []
                    for interval in intervals:
                        int_start, int_end, int_label = interval['Interval Start DT'], interval['Interval End DT'], interval['Time Intervals']; row = {'TIME': int_label}
                        int_moves = df_moves[(df_moves['timestamp'] >= int_start) & (df_moves['timestamp'] < int_end) & (df_moves['direction_from'] == from_dir_match)]
                        for to_dir_rep in to_dirs_rep:
                            to_dir_match = to_dir_rep.split("To ")[-1]; dir_total = 0
                            for vtype in all_valid_vehicle_types:
                                count = len(int_moves[(int_moves['direction_to'] == to_dir_match) & (int_moves['vehicle_type'] == vtype)])
                                row[f"{to_dir_rep}_{vtype}"] = count; dir_total += count
                            row[f"{to_dir_rep}_TOTAL"] = dir_total
                        sheet_data.append(row)
                    if not sheet_data: debug_print(f"Skipping sheet '{from_dir_rep}': No data."); continue
                    df_sheet = pd.DataFrame(sheet_data); cols_ord, col_ren = ['TIME'], {}
                    for to_dir_rep in to_dirs_rep:
                         for vtype in all_valid_vehicle_types: k=f"{to_dir_rep}_{vtype}"; cols_ord.append(k); col_ren[k]=vtype
                         k_tot=f"{to_dir_rep}_TOTAL"; cols_ord.append(k_tot); col_ren[k_tot]='TOTAL'
                    for col in cols_ord:
                        if col not in df_sheet.columns and col != 'TIME': df_sheet[col] = 0
                    df_sheet = df_sheet[cols_ord].rename(columns=col_ren)
                    df_sheet.to_excel(writer, sheet_name=from_dir_rep, index=False); debug_print(f"Created '{from_dir_rep}' sheet.")
            else: debug_print("No data for directional sheets (processed_movements or intervals empty).")

        print(f"Excel report generated: {EXCEL_REPORT_NAME}"); return EXCEL_REPORT_NAME
    except Exception as e: print(f"Error creating Excel report: {e}"); import traceback; traceback.print_exc(); return None