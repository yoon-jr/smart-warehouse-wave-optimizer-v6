# Data Attribution

## Dataset

**Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company**

- Contributor: Rodrigo Furlan de Assis
- Repository: Mendeley Data
- Version: 1
- Published: 24 December 2024
- DOI: `10.17632/pf2w725pw3.1`
- URL: `https://data.mendeley.com/datasets/pf2w725pw3/1`
- License: Creative Commons Attribution 4.0 International (`CC BY 4.0`)

## Related data article

R. Furlan de Assis, W. de Paula Ferreira, and M. Ouhimmou,
“Order picking dataset from a warehouse of a footwear manufacturing company,”
*Data in Brief*, vol. 61, 111837, 2025.

- DOI: `10.1016/j.dib.2025.111837`

## Transformations made by this project

This project does not redistribute the source dataset. The user downloads it directly from Mendeley Data.

The preprocessing code:

1. reads `Picking_Wave.csv` and `Storage_Location.csv`;
2. joins storage identifiers with Cartesian coordinates;
3. scales coordinates to a four-zone 20×20 simulator grid;
4. maps each pick line to one independent robot task;
5. spreads pick lines within a wave according to sequence and wave duration;
6. derives task service time from quantity;
7. creates a five-minute Zone demand training table;
8. records detected columns, time basis, assumptions, and join rate in `data/footwear_manifest.json`.

The transformation does not claim to reproduce continuous human-picker or robot trajectories.
