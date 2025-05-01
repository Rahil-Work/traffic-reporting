# project/reporting/excel_report.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os
from collections import OrderedDict

# --- Configuration Import ---
from config import VEHICLE_ID_MAP, REPORT_OUTPUT_DIR, VALID_MOVEMENTS
# --- Utility Import ---
from core.utils import debug_print, get_display_direction

# --- Helper Function: Build RSA Direction Map (Corrected Exit Order) ---
def _build_rsa_direction_map(primary_direction, active_directions, valid_movements_config):
    """Builds a map from (entry_key, exit_key) to RSA movement number."""
    rsa_movement_map = OrderedDict()
    movement_number = 1

    STANDARD_ORDER = ['north', 'west', 'south', 'east'] # Reference CCW order
    OPPOSITE = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}

    if not active_directions:
        debug_print("Error: No active directions provided for RSA map generation.")
        return rsa_movement_map

    if not primary_direction or primary_direction not in active_directions:
        primary_direction = active_directions[0]
        debug_print(f"Warning: Primary direction invalid/inactive. Using first active: {primary_direction}")

    try:
        start_index = STANDARD_ORDER.index(primary_direction)
    except ValueError: # Fallback if primary_direction isn't in standard order
        start_index = STANDARD_ORDER.index(active_directions[0]) if active_directions else 0

    ordered_active_entry_dirs = []
    processed_dirs = set()
    if active_directions:
        # Build the ordered list of active entry directions starting from primary
        for i in range(len(STANDARD_ORDER)):
            current_dir = STANDARD_ORDER[(start_index + i) % len(STANDARD_ORDER)]
            if current_dir in active_directions and current_dir not in processed_dirs:
                ordered_active_entry_dirs.append(current_dir)
                processed_dirs.add(current_dir)

    debug_print(f"RSA Numbering: Ordered Active Entry Dirs = {ordered_active_entry_dirs}")

    # Iterate through ordered entry dirs
    for entry_dir in ordered_active_entry_dirs:
        debug_print(f"  Processing Entry: {entry_dir}")
        try:
            entry_idx = STANDARD_ORDER.index(entry_dir)
        except ValueError:
            debug_print(f"    Skipping invalid entry_dir: {entry_dir}")
            continue

        # Determine relative directions based on the standard CCW order
        straight_dir = OPPOSITE.get(entry_dir)
        right_dir = STANDARD_ORDER[(entry_idx + 1) % len(STANDARD_ORDER)] # Next in CCW order is right
        left_dir = STANDARD_ORDER[(entry_idx - 1 + len(STANDARD_ORDER)) % len(STANDARD_ORDER)] # Previous is left

        # Define the relative exit order: Left -> Straight -> Right
        relative_exit_order = [left_dir, straight_dir, right_dir]
        debug_print(f"    Relative Exit Order Used: {relative_exit_order}")

        allowed_exits = valid_movements_config.get(entry_dir, [])
        processed_exits_for_entry = set() # Track exits processed for this entry

        # Iterate through relative exit order
        for exit_dir in relative_exit_order:
            # Check if this exit is valid
            is_valid = (
                exit_dir is not None and
                exit_dir in active_directions and      # Is it an active zone?
                exit_dir != entry_dir and             # Not a U-turn?
                exit_dir in allowed_exits and         # Allowed by config?
                exit_dir not in processed_exits_for_entry # Not already numbered for this entry?
            )

            if is_valid:
                if (entry_dir, exit_dir) not in rsa_movement_map:
                    rsa_movement_map[(entry_dir, exit_dir)] = movement_number
                    debug_print(f"    Mapping: {entry_dir} -> {exit_dir} = #{movement_number}")
                    movement_number += 1
                    processed_exits_for_entry.add(exit_dir) # Mark as processed

    debug_print(f"RSA Direction Map created with {len(rsa_movement_map)} entries: {rsa_movement_map}")
    return rsa_movement_map


# --- Main Report Function ---
def create_excel_report(completed_paths_data, start_datetime, primary_direction, video_path):
    """
    Creates a multi-sheet Excel report: Vehicle Logs (1 row/path), Intervals (24hr),
    Directional Counts (multi-header), RSA Output.
    """
    print(f"\n--- create_excel_report (Complete Code - Final Structure) ---")
    if not completed_paths_data:
         print("WARNING: No completed paths data received!")
         return None
    # print(f"Received {len(detection_events)} structured events (currently unused).")
    print(f"Received {len(completed_paths_data)} completed path summaries.")

    all_valid_vehicle_types = list(VEHICLE_ID_MAP.keys())
    if not all_valid_vehicle_types:
        print("ERROR: VEHICLE_ID_MAP in config appears empty.")
        return None

    # --- 1. Prepare Base Data ---
    valid_path_ids = set()
    valid_paths_structured = [] # For directional counts and RSA sheet
    vehicle_log_summary_data = [] # For Vehicle Logs sheet

    active_directions_set = set()
    if completed_paths_data:
        for path in completed_paths_data:
            v_id = path.get('id'); entry_t = path.get('entry_time'); exit_t = path.get('exit_time'); status = path.get('status')
            entry_dir = path.get('entry_direction'); exit_dir = path.get('exit_direction'); t_in_int = path.get('time_in_intersection')
            is_valid_entry_dir = entry_dir and entry_dir not in ['UNKNOWN', None]; is_valid_exit_dir = exit_dir and exit_dir not in ['UNKNOWN', 'TIMEOUT', 'FORCED', None, 'IN_PROGRESS', 'N/A']
            is_valid_status = status == 'exited'; has_valid_times = entry_t and exit_t and exit_t >= entry_t

            if is_valid_status and is_valid_entry_dir and is_valid_exit_dir and has_valid_times and v_id is not None:
                v_type = path.get('type', 'UnkType')
                if v_type not in all_valid_vehicle_types: continue
                valid_path_ids.add(v_id)
                entry_dir_disp = get_display_direction(entry_dir); exit_dir_disp = get_display_direction(exit_dir)
                entry_time_str = entry_t.strftime('%H:%M:%S.%f')[:-3]; exit_time_str = exit_t.strftime('%H:%M:%S.%f')[:-3]
                # Data for Vehicle Logs sheet
                vehicle_log_summary_data.append({
                    'Vehicle Tracker ID': v_id, 'Object ID': VEHICLE_ID_MAP.get(v_type, ''), 'Object Name': v_type,
                    'Entry Direction': entry_dir_disp, 'Exit Direction': exit_dir_disp,
                    'Entry Time': entry_time_str, 'Exit Time': exit_time_str,
                    'Time in Intersection (s)': round(t_in_int, 2) if t_in_int is not None else 'N/A' })
                # Data for other sheets (use raw keys for direction map lookup)
                valid_paths_structured.append({
                     'vehicle_type': v_type, 'direction_from': entry_dir, 'direction_to': exit_dir,
                     'timestamp': exit_t })
                active_directions_set.add(entry_dir); active_directions_set.add(exit_dir)
    debug_print(f"Processed {len(valid_path_ids)} valid completed paths.")
    active_directions = sorted(list(active_directions_set)) # Use only directions involved in valid paths
    debug_print(f"Active directions derived from valid paths: {active_directions}")

    # --- Build RSA Direction Map ---
    rsa_direction_map = _build_rsa_direction_map(primary_direction, active_directions, VALID_MOVEMENTS)

    # --- 2. Create Intervals (Full 24 Hours) ---
    intervals = []; report_date = start_datetime.date(); day_start_dt = datetime.combine(report_date, time.min); day_end_dt = day_start_dt + timedelta(days=1); curr_interval_start = day_start_dt
    interval_bins = [day_start_dt]; interval_labels = []
    while curr_interval_start < day_end_dt:
        interval_end = curr_interval_start + timedelta(minutes=15); interval_labels.append(f"{curr_interval_start.strftime('%H:%M:%S')} - {interval_end.strftime('%H:%M:%S')}")
        interval_bins.append(interval_end); intervals.append({'Date': curr_interval_start.strftime('%m/%d/%Y'), 'Time Intervals': interval_labels[-1], 'Start Time': curr_interval_start.strftime('%H:%M:%S'), 'End Time': interval_end.strftime('%H:%M:%S')}); curr_interval_start += timedelta(minutes=15)

    # --- 3. Prepare DataFrames ---
    df_log_summary = pd.DataFrame(vehicle_log_summary_data); df_intervals = pd.DataFrame(intervals)
    if not valid_paths_structured:
        df_moves = pd.DataFrame(columns=['vehicle_type', 'direction_from', 'direction_to', 'timestamp', 'TIME'])
    else:
        # Create display names needed for pivot table columns
        for path in valid_paths_structured:
            path['direction_from_disp'] = get_display_direction(path['direction_from'])
            path['direction_to_disp'] = get_display_direction(path['direction_to'])
        df_moves = pd.DataFrame(valid_paths_structured); df_moves['timestamp'] = pd.to_datetime(df_moves['timestamp'])
        time_cat = pd.CategoricalDtype(categories=interval_labels, ordered=True); df_moves['TIME'] = pd.cut(df_moves['timestamp'], bins=interval_bins, labels=interval_labels, right=False, ordered=True).astype(time_cat)
        vehicle_cat = pd.CategoricalDtype(categories=all_valid_vehicle_types, ordered=False); df_moves['vehicle_type'] = df_moves['vehicle_type'].astype(vehicle_cat)

    # --- Create Excel File ---
    excel_created_path = None
    try:
        output_report_dir = REPORT_OUTPUT_DIR
        os.makedirs(output_report_dir, exist_ok=True)
        output_filename_base=f"detection_logs_{os.path.splitext(os.path.basename(video_path))[0]}"
        excel_full_path = os.path.join(output_report_dir, f"{output_filename_base}.csv")


        with pd.ExcelWriter(excel_full_path, engine='xlsxwriter') as writer:
            workbook = writer.book

            # --- Sheet: Vehicle Logs ---
            if not df_log_summary.empty:
                 log_cols = ['Vehicle Tracker ID', 'Object ID', 'Object Name', 'Entry Direction', 'Exit Direction', 'Entry Time', 'Exit Time', 'Time in Intersection (s)']
                 df_logs = df_log_summary[[col for col in log_cols if col in df_log_summary.columns]]
                 df_logs.to_excel(writer, sheet_name='Vehicle Logs', index=False)
                 debug_print(f"Writing 'Vehicle Logs' sheet ({len(df_logs)} rows).")
            else: debug_print("No data for 'Vehicle Logs' sheet.")

            # --- Sheet: Intervals ---
            if not df_intervals.empty:
                df_intervals[['Date', 'Time Intervals', 'Start Time', 'End Time']].to_excel(writer, sheet_name='Intervals', index=False)
                debug_print(f"Writing 'Intervals' sheet ({len(df_intervals)} rows).")
            else: debug_print("No data for 'Intervals' sheet.")

            # --- Sheets: Directional Counts ---
            if not df_moves.empty and not df_intervals.empty and active_directions:
                # Define formats
                header_format_main = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
                header_format_sub = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
                col_header_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'rotation': 90, 'border': 1})
                time_col_format = workbook.add_format({'align': 'left'})
                data_format = workbook.add_format({'align': 'center'})

                for from_dir_key in active_directions:
                    from_dir_sheet_name = f"From {from_dir_key.capitalize()}"
                    from_dir_match_display = from_dir_key.capitalize()
                    worksheet = writer.book.add_worksheet(from_dir_sheet_name)
                    writer.sheets[from_dir_sheet_name] = worksheet
                    debug_print(f"  Generating sheet: {from_dir_sheet_name}")

                    df_from = df_moves[df_moves['direction_from_disp'] == from_dir_match_display].copy()
                    possible_to_dirs_keys = sorted([d for d in active_directions if d != from_dir_key])
                    if not possible_to_dirs_keys: continue

                    # Pivot, Reindex, Calculate Totals
                    pivot_df = pd.pivot_table(df_from, index='TIME', columns=['direction_to_disp', 'vehicle_type'], aggfunc='size', fill_value=0, observed=False, dropna=False)
                    all_dest_display = [get_display_direction(d) for d in possible_to_dirs_keys]
                    multi_cols = pd.MultiIndex.from_product([all_dest_display, all_valid_vehicle_types], names=['direction_to', 'vehicle_type'])
                    pivot_df = pivot_df.reindex(columns=multi_cols, fill_value=0); pivot_df = pivot_df.reindex(interval_labels, fill_value=0)
                    for to_dir_disp in all_dest_display:
                        vehicle_cols_tuples = [(to_dir_disp, vtype) for vtype in all_valid_vehicle_types]; cols_to_sum = [col for col in vehicle_cols_tuples if col in pivot_df.columns]
                        if cols_to_sum: pivot_df[(to_dir_disp, 'TOTAL')] = pivot_df[cols_to_sum].sum(axis=1)
                        else: pivot_df[(to_dir_disp, 'TOTAL')] = 0
                    final_col_order = [];
                    for to_dir_disp in all_dest_display:
                        for vtype in all_valid_vehicle_types: final_col_order.append((to_dir_disp, vtype)); final_col_order.append((to_dir_disp, 'TOTAL'))
                    final_col_order = [col for col in final_col_order if col in pivot_df.columns];
                    if not final_col_order: continue
                    df_sheet = pivot_df[final_col_order].copy(); df_sheet.index.name = 'TIME'; df_sheet.reset_index(inplace=True)

                    # Write Headers Manually
                    num_vehicle_types = len(all_valid_vehicle_types); cols_per_dest = num_vehicle_types + 1; last_data_col_index = 0
                    if possible_to_dirs_keys: last_data_col_index = (cols_per_dest * len(possible_to_dirs_keys))
                    worksheet.merge_range(0, 0, 0, last_data_col_index, from_dir_sheet_name, header_format_main); worksheet.write(2, 0, "TIME", col_header_format)
                    current_col_idx = 1; header_col_tuples_written = []
                    for to_dir_key in possible_to_dirs_keys:
                        to_dir_display_hdr = f"To {to_dir_key.capitalize()}"; start_merge_col_idx = current_col_idx; num_cols_this_dest = 0
                        for vtype in all_valid_vehicle_types:
                            col_tuple = (get_display_direction(to_dir_key), vtype)
                            if col_tuple in final_col_order: worksheet.write(2, current_col_idx, vtype, col_header_format); header_col_tuples_written.append(col_tuple); current_col_idx += 1; num_cols_this_dest += 1
                        total_tuple = (get_display_direction(to_dir_key), 'TOTAL')
                        if total_tuple in final_col_order: worksheet.write(2, current_col_idx, "TOTAL", col_header_format); header_col_tuples_written.append(total_tuple); current_col_idx += 1; num_cols_this_dest += 1
                        if num_cols_this_dest > 0: worksheet.merge_range(1, start_merge_col_idx, 1, current_col_idx - 1, to_dir_display_hdr, header_format_sub)

                    # Write Data Manually with Explicit Casting
                    start_row = 3
                    for row_idx, time_label in enumerate(df_sheet['TIME']):
                        try: worksheet.write(start_row + row_idx, 0, str(time_label))
                        except Exception: worksheet.write(start_row + row_idx, 0, "Write Error")
                        for col_idx, col_tuple in enumerate(header_col_tuples_written):
                             try: value = df_sheet.loc[row_idx, col_tuple]; worksheet.write(start_row + row_idx, col_idx + 1, int(value))
                             except KeyError: worksheet.write(start_row + row_idx, col_idx + 1, 0)
                             except (ValueError, TypeError): worksheet.write(start_row + row_idx, col_idx + 1, 0)

                    # Apply Formatting
                    worksheet.set_column(0, 0, 18, time_col_format)
                    current_col_idx = 1
                    for col_tuple in header_col_tuples_written: col_width = 10 if col_tuple[1] == 'TOTAL' else 5; worksheet.set_column(current_col_idx, current_col_idx, col_width, data_format); current_col_idx += 1
                    debug_print(f"  Writing '{from_dir_sheet_name}' sheet ({len(df_sheet)} rows).")

            # Condition checks
            elif df_moves.empty: debug_print("Skipping directional sheets: No valid movements.")
            elif not intervals: debug_print("Skipping directional sheets: Intervals list empty.")
            elif not active_directions: debug_print("Skipping directional sheets: No active directions.")

            # --- Sheet: RSA Format Output ---
            debug_print(f"Checking conditions for RSA sheet: Map size={len(rsa_direction_map)}, Valid paths count={len(valid_paths_structured)}")
            if rsa_direction_map and valid_paths_structured:
                debug_print("Generating RSA Format Output sheet...")
                rsa_sheet_name = "RSA Output"
                rsa_worksheet = writer.book.add_worksheet(rsa_sheet_name)
                writer.sheets[rsa_sheet_name] = rsa_worksheet

                # Write Header
                L0_val_1 = len(rsa_direction_map); L0_val_2 = len(rsa_direction_map); L0_val_3 = len(active_directions)
                rsa_header = [
                    "H0,1,320,3,RSA Standard Format Version 3.20",
                    f"S0,ME055,ME055, Intersection Name Placeholder,-31.365299,29.572933", # Example
                    f"I0,00001,Emaan_Traffic_Capture_Application", "D0,M,L",
                    f"D1,{start_datetime.strftime('%y%m%d')},0000000,{start_datetime.strftime('%y%m%d')},235959999,{start_datetime.strftime('%y%m%d')},0000000",
                    f"L0,{L0_val_1},{L0_val_2},{L0_val_3}" ]
                for r, row_str in enumerate(rsa_header): rsa_worksheet.write_string(r, 0, row_str)

                # Prepare and Write Data Lines
                rsa_data_lines = []; valid_paths_structured.sort(key=lambda x: x['timestamp'])
                for path in valid_paths_structured:
                    entry_key = path['direction_from']; exit_key = path['direction_to']; ts_obj = path['timestamp']; v_type = path['vehicle_type']
                    dir_num = rsa_direction_map.get((entry_key, exit_key), 0)
                    base_id = VEHICLE_ID_MAP.get(v_type, '?')
                    v_code = f"{base_id},1" if ',1' not in base_id else base_id
                    ts_str = ts_obj.strftime('%y%m%d,%H%M%S%f')[:-3]
                    rsa_line = f"10,9,1,0,{ts_str},{dir_num},{dir_num},1,{v_code}"
                    rsa_data_lines.append(rsa_line)
                start_data_row = len(rsa_header)
                for r, row_str in enumerate(rsa_data_lines): rsa_worksheet.write_string(start_data_row + r, 0, row_str)
                rsa_worksheet.set_column(0, 0, 60); debug_print(f"Writing 'RSA Output' sheet ({len(rsa_data_lines)} rows).")

            elif not rsa_direction_map: debug_print("Skipping 'RSA Output' sheet: RSA Direction map is empty.")
            else: debug_print("Skipping 'RSA Output' sheet: No valid paths processed.")

        print(f"Excel report generated: {excel_full_path}")
        excel_created_path = excel_full_path

    except Exception as e:
        print(f"Error creating Excel report: {e}")
        import traceback; traceback.print_exc()
        return None

    return excel_created_path