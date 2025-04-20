# project/reporting/excel_report.py
import pandas as pd
import re
import numpy as np # Added for potential use with zone_tracker keys if needed
from datetime import datetime, timedelta
from config import (
    EXCEL_REPORT_NAME, VEHICLE_ID_MAP
)
from core.utils import debug_print, get_display_direction


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

    # Pattern 2: Exit/Timeout/Forced "VEHICLE TYPE {ACTION} [TO {DIR_TO}] [(FROM {DIR_FROM})]"
    # Handles cases like TIMED OUT where "TO DIR" might be absent
    match_exit = re.match(r"^(.*?)\s+(EXITED\s+TO|TIMED\s+OUT|FORCED\s+EXIT)\s*([^\(]+?)?(\s+\(FROM\s+(\w+)\))?$", event_str, re.IGNORECASE)
    if match_exit:
        vehicle_type=match_exit.group(1).strip(); action_keyword=match_exit.group(2).strip().upper()
        # Group 3 captures the direction_to part, might be None for TIMED OUT etc.
        direction_to_raw = match_exit.group(3).strip().upper() if match_exit.group(3) else None
        direction_from = match_exit.group(5).strip().upper() if match_exit.group(5) else None

        if "EXITED TO" in action_keyword:
            event_action = "EXIT"
            direction_to = direction_to_raw # Use the parsed TO direction
        elif "TIMED OUT" in action_keyword:
            event_action = "TIMEOUT"
            direction_to = "TIMEOUT" # Standardize direction_to
        elif "FORCED EXIT" in action_keyword:
            event_action = "FORCED_EXIT"
            direction_to = "FORCED" # Standardize direction_to
        else:
            event_action = "UNKNOWN" # Should not happen

        # If it was an exit but TO direction wasn't explicitly parsed (unlikely with current regex)
        if event_action == "EXIT" and direction_to is None:
             direction_to = "UNKNOWN_EXIT_TARGET"

        return vehicle_type, event_action, direction_to, direction_from

    debug_print(f"Could not parse event string with regex: '{event_str}'");
    return None, "UNKNOWN", None, None


# --- Main Report Function (Updated) ---
def create_excel_report(detection_events, completed_paths_data, start_datetime, zone_tracker=None):
    """
    Creates a comprehensive Excel report. Includes valid entry/exit pairs
    in the 'Complete Paths' sheet and dynamically generates directional sheets
    based on the zones present in the zone_tracker.
    Args:
        detection_events (list): List of dictionaries for ENTRY/EXIT/etc. logs.
        completed_paths_data (list): List of dictionaries from VehicleTracker.
        start_datetime (datetime): The starting timestamp for interval calculations.
        zone_tracker (ZoneTracker, optional): The zone tracker instance containing active zones.
    """
    print(f"\n--- create_excel_report ---")
    print(f"Received {len(detection_events)} detection events for logs.")
    print(f"Received {len(completed_paths_data)} completed path summaries from tracker.")
    if not detection_events and not completed_paths_data:
         print("WARNING: No detection events or completed paths received!")
         return None

    all_valid_vehicle_types = list(VEHICLE_ID_MAP.keys())
    vehicle_logs_data = []
    processed_movements = []
    complete_paths_report_data = []
    vehicle_entry_info = {}

    # --- Process Events for Vehicle Logs and Directional Counts ---
    for event in detection_events:
        event_str = event.get('detection', ''); timestamp_str_ms = event.get('timestamp', '')
        vehicle_id = event.get('vehicle_id', 'UnknownID'); status = event.get('status', 'unknown')

        try: # Reconstruct timestamp
            if len(timestamp_str_ms)>6: time_part,ms_part=timestamp_str_ms[:6],timestamp_str_ms[6:]; base_time=datetime.strptime(time_part,"%H%M%S").time(); microseconds=int(ms_part.ljust(3,'0'))*1000; event_time_part=base_time.replace(microsecond=microseconds)
            else: event_time_part=datetime.strptime(timestamp_str_ms,"%H%M%S").time()
            event_timestamp = datetime.combine(start_datetime.date(), event_time_part)
        except ValueError: debug_print(f"Skip Log: Bad timestamp '{timestamp_str_ms}'"); continue

        vehicle_type, event_action, direction_to, direction_from = parse_detection_string(event_str)
        if event_action == "UNKNOWN" or vehicle_type is None: continue

        direction_to_log = direction_to; direction_from_log = direction_from
        if event_action == "ENTRY":
            vehicle_entry_info[vehicle_id] = {'type': vehicle_type, 'entry_dir': direction_from, 'entry_time': event_timestamp}
            direction_to_log = 'IN_PROGRESS'
        elif event_action in ["EXIT", "TIMEOUT", "FORCED_EXIT"]:
             if direction_from is None: direction_from = vehicle_entry_info.get(vehicle_id, {}).get('entry_dir', 'UNKNOWN')
             direction_from_log = direction_from
             if event_action == "EXIT" and direction_to not in ['UNKNOWN', 'TIMEOUT', 'FORCED', 'IN_PROGRESS', 'UNKNOWN_EXIT_TARGET', None]: # Ensure valid exit target
                 entry_info_mov = vehicle_entry_info.get(vehicle_id)
                 if entry_info_mov and entry_info_mov['entry_dir'] == direction_from:
                     processed_movements.append({
                         'vehicle_type': vehicle_type,
                         'direction_from': get_display_direction(direction_from), # Store display version for easier filtering later
                         'direction_to': get_display_direction(direction_to),     # Store display version
                         'timestamp': event_timestamp })

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

    # --- Prepare FILTERED Complete Paths Data ---
    valid_complete_paths_count = 0
    if completed_paths_data:
        for path in completed_paths_data:
            entry_t=path.get('entry_time'); exit_t=path.get('exit_time'); status=path.get('status')
            entry_dir=path.get('entry_direction'); exit_dir=path.get('exit_direction'); t_in_int=path.get('time_in_intersection')
            is_valid_entry_dir=entry_dir and entry_dir not in ['UNKNOWN', None]
            is_valid_exit_dir=exit_dir and exit_dir not in ['UNKNOWN', 'TIMEOUT', 'FORCED', None]
            if status == 'exited' and is_valid_entry_dir and is_valid_exit_dir and entry_t and exit_t:
                valid_complete_paths_count += 1; v_type = path.get('type', 'UnkType')
                entry_dir_disp=get_display_direction(entry_dir); exit_dir_disp=get_display_direction(exit_dir)
                entry_time_str=entry_t.strftime('%H:%M:%S.%f')[:-3]; exit_time_str=exit_t.strftime('%H:%M:%S.%f')[:-3]
                complete_path_str=f"{entry_dir_disp} TO {exit_dir_disp}"
                complete_paths_report_data.append({
                    'Vehicle Type': v_type, 'Entry Direction': entry_dir_disp, 'Exit Direction': exit_dir_disp,
                    'Entry Time': entry_time_str, 'Exit Time': exit_time_str,
                    'Time in Intersection (s)': round(t_in_int, 2) if t_in_int is not None else 'N/A',
                    'Complete Path': complete_path_str, 'Status': status })
    total_completed = len(completed_paths_data) if completed_paths_data else 0
    debug_print(f"Filtered complete paths: Included {valid_complete_paths_count} valid paths out of {total_completed} total summaries.")

    # --- Create Excel File ---
    try:
        with pd.ExcelWriter(EXCEL_REPORT_NAME, engine='xlsxwriter') as writer:
            # Sheet: Vehicle Logs
            if vehicle_logs_data:
                 log_cols = ['Timestamp','Date','Time','Object ID','Object Name','Event','Direction From','Direction To','Status','Vehicle Tracker ID']
                 df_logs = pd.DataFrame(vehicle_logs_data)[log_cols]; df_logs.to_excel(writer, sheet_name='Vehicle Logs', index=False); debug_print(f"Created 'Vehicle Logs' sheet ({len(df_logs)} rows).")
            else: debug_print("No data for 'Vehicle Logs' sheet.")

            # Sheet: Complete Paths
            if complete_paths_report_data:
                path_cols = ['Vehicle Type','Entry Direction','Exit Direction','Entry Time','Exit Time','Time in Intersection (s)','Complete Path','Status']
                df_complete = pd.DataFrame(complete_paths_report_data)[path_cols]; df_complete.to_excel(writer, sheet_name='Complete Paths', index=False); debug_print(f"Created 'Complete Paths' sheet ({len(df_complete)} rows).")
            else: debug_print("No data for 'Complete Paths' sheet (after filtering).")

            # Sheet: Intervals
            intervals = []; report_dur_hrs = 7; end_limit = start_datetime + timedelta(hours=report_dur_hrs); curr_interval_start = start_datetime
            while curr_interval_start < end_limit:
                interval_end = min(curr_interval_start + timedelta(minutes=15), end_limit); intervals.append({'Date': curr_interval_start.strftime('%m/%d/%Y'), 'Time Intervals': f"{curr_interval_start.strftime('%H:%M:%S')} - {interval_end.strftime('%H:%M:%S')}", 'Start Time': curr_interval_start.strftime('%H:%M:%S'), 'End Time': interval_end.strftime('%H:%M:%S'), 'Interval Start DT': curr_interval_start, 'Interval End DT': interval_end}); curr_interval_start += timedelta(minutes=15)
            if intervals: df_intervals = pd.DataFrame(intervals); df_intervals[['Date', 'Time Intervals', 'Start Time', 'End Time']].to_excel(writer, sheet_name='Intervals', index=False); debug_print("Created 'Intervals' sheet.")
            else: debug_print("No data for 'Intervals' sheet.")

            # Sheets: Directional Counts (Dynamically generated)
            active_directions = []
            if zone_tracker and hasattr(zone_tracker, 'zones') and zone_tracker.zones:
                 active_directions = list(zone_tracker.zones.keys()) # e.g., ['north', 'west', 'south']
                 print(f"Report: Generating directional sheets for active zones: {active_directions}")
            else: print("Report: Warning - ZoneTracker object not provided or has no zones. Skipping dynamic directional sheets.")

            if processed_movements and intervals and active_directions:
                debug_print(f"Processing {len(processed_movements)} movements for directional sheets...")
                # Convert processed_movements (which used get_display_direction) to DataFrame
                df_moves = pd.DataFrame(processed_movements)

                for from_dir_key in active_directions:
                    from_dir_sheet_name = f"From {from_dir_key.capitalize()}"
                    # Use the Display Name (e.g., 'North') for filtering the DataFrame column
                    from_dir_match_display = from_dir_key.capitalize()

                    sheet_data = []
                    possible_to_dirs = [d for d in active_directions if d != from_dir_key]

                    for interval in intervals:
                        int_start, int_end = interval['Interval Start DT'], interval['Interval End DT']
                        int_label = interval['Time Intervals']
                        row = {'TIME': int_label}

                        # Filter movements for interval and the specific 'from' direction (using display name)
                        int_moves = df_moves[
                            (df_moves['timestamp'] >= int_start) &
                            (df_moves['timestamp'] < int_end) &
                            (df_moves['direction_from'] == from_dir_match_display) # Match display name
                        ]

                        for to_dir_key in possible_to_dirs:
                            # Use Display Name for header prefix and filtering
                            to_dir_display = to_dir_key.capitalize()
                            to_dir_header_prefix = f"To {to_dir_display}"

                            dir_total = 0
                            for vtype in all_valid_vehicle_types:
                                # Filter using display name for 'to' direction
                                count = len(int_moves[
                                    (int_moves['direction_to'] == to_dir_display) &
                                    (int_moves['vehicle_type'] == vtype)
                                ])
                                row[f"{to_dir_header_prefix}_{vtype}"] = count
                                dir_total += count
                            row[f"{to_dir_header_prefix}_TOTAL"] = dir_total
                        sheet_data.append(row)

                    if not sheet_data:
                        debug_print(f"Skipping sheet '{from_dir_sheet_name}': No data for this direction.")
                        continue

                    df_sheet = pd.DataFrame(sheet_data)
                    cols_ord, col_ren = ['TIME'], {}

                    # Build columns based on possible 'to' directions (using display names)
                    for to_dir_key in possible_to_dirs:
                        to_dir_display = to_dir_key.capitalize()
                        to_dir_header_prefix = f"To {to_dir_display}"
                        for vtype in all_valid_vehicle_types:
                            k = f"{to_dir_header_prefix}_{vtype}"
                            cols_ord.append(k); col_ren[k] = vtype # Use vehicle type as final column name
                        k_tot = f"{to_dir_header_prefix}_TOTAL"
                        cols_ord.append(k_tot); col_ren[k_tot] = 'TOTAL'

                    # Ensure all expected columns exist, fill with 0 if missing
                    for col in cols_ord:
                        if col not in df_sheet.columns and col != 'TIME':
                            df_sheet[col] = 0

                    # Reorder and rename columns
                    df_sheet = df_sheet[cols_ord].rename(columns=col_ren)

                    # Write the sheet
                    df_sheet.to_excel(writer, sheet_name=from_dir_sheet_name, index=False)
                    debug_print(f"Created '{from_dir_sheet_name}' sheet.")
            elif not active_directions:
                 debug_print("Skipping directional sheets because no active zones were identified.")
            else: # No movements or intervals
                debug_print("No data/intervals/movements for directional sheets.")

        print(f"Excel report generated: {EXCEL_REPORT_NAME}")
        return EXCEL_REPORT_NAME
    except Exception as e:
        print(f"Error creating Excel report: {e}")
        import traceback; traceback.print_exc()
        return None