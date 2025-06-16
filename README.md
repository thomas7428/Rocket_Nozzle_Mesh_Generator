# Rocket Engine Mesh Generator

This project provides a Python-based pipeline for generating 2D and 3D mesh geometries of rocket engine nozzles, with support for real or custom geometry data, cooling channels, and reinforcement ribs. The pipeline is designed for use with the GMSH Python API and is suitable for CFD, FEA, or other simulation workflows.

## Features
- Load nozzle geometry from CSV or custom parameters
- Generate smooth nozzle profiles (Bézier, de Laval, etc.)
- Add cooling channels and reinforcement ribs
- Create 3D geometry by revolving 2D profiles
- Export meshes to `.msh` and `.stl` formats
- Visualize profiles and geometry

## Folder Structure
- `mesh_generator_v0_2/` — Main pipeline (notebooks, scripts, mesh outputs)
- `Python_version/` — Standalone Python scripts (including CSV profile generator)
- `Rocket_Engine_Meshes/` — Output meshes

## Requirements
- Python 3.8+
- `gmsh`, `numpy`, `matplotlib`, `scipy`, `pandas`

## Quick Start
1. Place your geometry CSV files in the working directory.
   - **If you do not have a suitable profile CSV file, you can generate one using the CSV profile generator script:**
     ```bash
     python Python_version/csv_profiles_bezier_generator.py
     ```
   - Follow the prompts or edit the script to define your custom profile.
2. Run the main script:
   ```bash
   python mesh_generator_v0_2/mesh_generator.py
   ```
3. Mesh files will be saved in `Rocket_Engine_Meshes/`.

## Jupyter Notebook Version (Recommended)
A Jupyter Notebook version of the pipeline is available in the `mesh_generator_v0_2/` folder. **It is recommended to use the notebook version for a more interactive, visual, and beginner-friendly experience.** The notebook allows you to:
- Visualize each step and geometry interactively
- Modify parameters and rerun cells easily
- Debug and explore results step by step

## Customization
- Edit parameters in the script or provide your own CSV files for custom geometries.
- Adjust mesh resolution and features (channels, reinforcements) as needed.

## Troubleshooting
- Ensure all dependencies are installed.
- Check your input data for completeness and correctness.
- Review script output for error messages.

## License
MIT License

---
For more details, see the notebook and script documentation in the repository.
