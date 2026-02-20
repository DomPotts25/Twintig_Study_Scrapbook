"""This file acts as the main module for this script."""

import csv
import os
import traceback

import adsk.core
import adsk.fusion

app = adsk.core.Application.get()
ui = app.userInterface


def run(context):
    ui = None
    try:
        # === SETUP ===
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        rootComp = design.rootComponent

        # File paths
        csv_path = "C:/Users/dm-potts-admin/Documents/Postdoc/UWE/Outside_Interactions/TwinTig_Hardware/Ring_Design/Dom/ring_output_Dom.csv"
        output_folder = "C:/Users/dm-potts-admin/Documents/Postdoc/UWE/Outside_Interactions/TwinTig_Hardware/Ring_Design/Dom/Models_v4/"

        # Ensure output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Parameters to read
        param_columns = ["Ring_Width", "Ring_Height"]

        export_mgr = design.exportManager

        # Read CSV
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            count = 0
            for row in reader:
                count += 1

                ring_name = row.get("Ring_Name", "").strip()
                if not ring_name:
                    ui.messageBox(f"Skipping row {count}: No Ring_Name found.")
                    continue

                # Set parameters
                for param_name in param_columns:
                    if param_name not in row:
                        ui.messageBox(f"Missing parameter '{param_name}' in CSV. Row {count}")
                        continue

                    val = row[param_name].strip()
                    try:
                        if param_name == "Ring_Width":
                            val = str(float(val) + 0.0001)

                        param = design.userParameters.itemByName(param_name)
                        if param:
                            param.expression = val
                        else:
                            ui.messageBox(f"Parameter '{param_name}' not found in design.")
                    except Exception as e:
                        ui.messageBox(f"Error setting parameter '{param_name}' to '{val}'\n{str(e)}")

                # --- Update Sketch Texts ---
                try:
                    sketches = rootComp.sketches

                    # Width Text
                    width_sketch = sketches.itemByName("Width_Text")
                    if width_sketch:
                        for sketch_text in width_sketch.sketchTexts:
                            sketch_text.text = row.get("Ring_Width_Name", "")

                    # Height Text
                    height_sketch = sketches.itemByName("Height_Text")
                    if height_sketch:
                        for sketch_text in height_sketch.sketchTexts:
                            sketch_text.text = row.get("Ring_Height_Name", "")

                except Exception as e:
                    ui.messageBox(f"Error updating sketch texts for row {count}: {str(e)}")

                design.computeAll()

                # Export STEP
                step_path = os.path.join(output_folder, f"{ring_name}.step")
                step_options = export_mgr.createSTEPExportOptions(step_path, rootComp)
                export_mgr.execute(step_options)

                # Export Fusion Archive (F3D)
                # f3d_path = os.path.join(output_folder, f'{ring_name}.f3d')
                # f3d_options = export_mgr.createFusionArchiveExportOptions(f3d_path)
                # export_mgr.execute(f3d_options)

        ui.messageBox(f"Finished exporting {count} models to STEP and F3D.")

    except Exception as e:
        if ui:
            ui.messageBox("Script failed:\n{}".format(traceback.format_exc()))
