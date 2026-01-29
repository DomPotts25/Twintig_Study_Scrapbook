import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os

print("Starting script...")

# Load master size lookup
lookup_csv = "Ring_Lookup_Main.csv"
print(f"Loading lookup CSV from: {lookup_csv}")
lookup_df = pd.read_csv(lookup_csv)
valid_sizes = lookup_df['Ring_Width_Name'].tolist()
print(f"Loaded {len(valid_sizes)} valid sizes")

# Use actual order from the master CSV
all_ring_sizes = valid_sizes.copy()

# All finger segments (left and right)
finger_segments = [
    f"{h}{f}" for h in ['L', 'R'] for f in [
        'P0', 'P1', 'P2', 'R0', 'R1', 'R2', 'M0', 'M1', 'M2',
        'I0', 'I1', 'I2', 'T0', 'T1']
]
print(f"Configured {len(finger_segments)} finger segments")

segment_data = {}
current_index = 0

# GUI Setup
root = tk.Tk()
root.title("Ring Size Entry")
root.geometry("500x350+100+100")

main_frame = tk.Frame(root)
main_frame.pack(pady=20)

info_label = tk.Label(main_frame, text="User Name:")
info_label.grid(row=0, column=0)
user_entry = tk.Entry(main_frame)
user_entry.grid(row=0, column=1)

segment_label = tk.Label(main_frame, text="", font=("Helvetica", 16, "bold"))
segment_label.grid(row=1, column=0, columnspan=2, pady=(20, 10))

size_label = tk.Label(main_frame, text="Ring Width Name:")
size_label.grid(row=2, column=0)
size_entry = tk.Entry(main_frame)
size_entry.grid(row=2, column=1)

before_label = tk.Label(main_frame, text="Variations Wide:")
before_label.grid(row=3, column=0)
before_entry = tk.Entry(main_frame)
before_entry.insert(0, "0")
before_entry.grid(row=3, column=1)

after_label = tk.Label(main_frame, text="Variations Narrow:")
after_label.grid(row=4, column=0)
after_entry = tk.Entry(main_frame)
after_entry.insert(0, "0")
after_entry.grid(row=4, column=1)

def get_next_segment():
    global current_index
    if current_index >= len(finger_segments):
        generate_output()
        return

    seg = finger_segments[current_index]
    segment_label.config(text=f"Enter data for: {seg}")
    size_entry.delete(0, tk.END)
    before_entry.delete(0, tk.END)
    after_entry.delete(0, tk.END)
    before_entry.insert(0, "0")
    after_entry.insert(0, "0")

def confirm_segment():
    global current_index
    seg = finger_segments[current_index]
    size = size_entry.get().strip().upper()
    try:
        before = int(before_entry.get())
        after = int(after_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Variation values must be integers.")
        return

    if size:
        segment_data[seg] = {
            'width': size,
            'before': before,
            'after': after
        }
    current_index += 1
    get_next_segment()

def skip_segment():
    global current_index
    current_index += 1
    get_next_segment()

def get_variations(size, before, after):
    if size not in all_ring_sizes:
        return []
    idx = all_ring_sizes.index(size)
    return all_ring_sizes[max(0, idx - before): idx + after + 1]

def generate_output():
    username = user_entry.get().strip()
    if not username:
        messagebox.showerror("Input Error", "User name is required.")
        return

    # Create output directory
    output_dir = os.path.join(os.getcwd(), username)
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []
    unique_rows = set()
    wide_rows = set()
    narrow_rows = set()

    for seg, data in segment_data.items():
        width_name = data['width']
        height_names = get_variations(width_name, data['before'], data['after'])
        width_row = lookup_df[lookup_df['Ring_Width_Name'] == width_name]
        if width_row.empty:
            continue
        width_value = width_row.iloc[0]['Ring_Width']
        for hname in height_names:
            height_row = lookup_df[lookup_df['Ring_Width_Name'] == hname]
            if height_row.empty:
                continue
            height_value = height_row.iloc[0]['Ring_Width']
            all_rows.append({
                'Segment': seg,
                'Ring_Name': f"{width_name}_{hname}",
                'Ring_Width_Name': width_name,
                'Ring_Height_Name': hname,
                'Ring_Width': width_value,
                'Ring_Height': height_value
            })
            ring_tuple = (f"{width_name}_{hname}", width_name, hname, width_value, height_value)
            unique_rows.add(ring_tuple)
            if height_value <= width_value:
                wide_rows.add(ring_tuple)
            else:
                narrow_rows.add(ring_tuple)

    if not all_rows:
        messagebox.showwarning("No Output", "No valid sizes were processed.")
        return

    final_df = pd.DataFrame(all_rows)
    out_path = os.path.join(output_dir, f"ring_output_{username}.csv")
    final_df.to_csv(out_path, index=False)

    unique_df = pd.DataFrame(list(unique_rows), columns=[
        'Ring_Name', 'Ring_Width_Name', 'Ring_Height_Name', 'Ring_Width', 'Ring_Height'])
    unique_df.to_csv(os.path.join(output_dir, f"ring_output_unique_{username}.csv"), index=False)

    wide_df = pd.DataFrame(list(wide_rows), columns=[
        'Ring_Name', 'Ring_Width_Name', 'Ring_Height_Name', 'Ring_Width', 'Ring_Height'])
    wide_df.to_csv(os.path.join(output_dir, f"ring_output_wide_{username}.csv"), index=False)

    narrow_df = pd.DataFrame(list(narrow_rows), columns=[
        'Ring_Name', 'Ring_Width_Name', 'Ring_Height_Name', 'Ring_Width', 'Ring_Height'])
    narrow_df.to_csv(os.path.join(output_dir, f"ring_output_narrow_{username}.csv"), index=False)

    messagebox.showinfo("Success", f"Output saved in folder: {output_dir}")
    root.destroy()

confirm_btn = tk.Button(main_frame, text="Confirm", command=confirm_segment, height=2, width=15, bg="lightblue")
confirm_btn.grid(row=5, column=0, pady=10)

skip_btn = tk.Button(main_frame, text="Skip", command=skip_segment, height=1, width=10)
skip_btn.grid(row=5, column=1, pady=10)

get_next_segment()

print("Launching main loop...")
root.mainloop()
print("Main loop exited.")