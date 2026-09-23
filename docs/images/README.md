# Report plot assets

These PNG files are the original embedded images extracted from `Report_DTB_Game.pdf`. They retain the report's native resolution, labels, data, and colors. No plots were redrawn or rescaled. The report PDF itself is not included in this project.

| Image | Report figure | Report page |
| --- | --- | --- |
| [Cournot step-size diagnostics](cournot-step-size.png) | 3 | 10 |
| [Cournot Neural-DTB clouds](cournot-dtb.png) | 4 | 10 |
| [Cournot explicit Euler clouds](cournot-euler.png) | 5 | 10 |
| [Oscillatory frequency diagnostics](oscillatory-diagnostics.png) | 6 | 12 |
| [Oscillatory final clouds](oscillatory-clouds.png) | 7 | 12 |
| [Eight-dimensional MMNN-DTB clouds](potential-8d-dtb.png) | 9 | 14 |
| [Eight-dimensional RK4 clouds](potential-8d-rk4.png) | 10 | 14 |

The Cournot images show the deterministic mode. The oscillatory report configuration uses h=0.001 and SVD cutoff 1e-8. These report presets differ from some supplied notebook defaults; the images are historical report results, not a claim that the current defaults reproduced them.
