# Third-party training data

Public datasets used for pretraining the spine detector (plan §0, §3). All
are CC BY 4.0 — commercial use permitted, attribution required.

| Dataset | Source | License | Local path |
|---|---|---|---|
| Book Spine Segmentation | [leo-ueno/book-spine-segmentation](https://universe.roboflow.com/leo-ueno/book-spine-segmentation) | CC BY 4.0 | `data/raw/leo-ueno-book-spine-segmentation/` |
| Book Spine Instance Segmentation | [harald-varner-xv5u7/book-spine-instance-segmentation](https://universe.roboflow.com/harald-varner-xv5u7/book-spine-instance-segmentation) | CC BY 4.0 | `data/raw/harald-varner-book-spine-instance-segmentation/` |
| DAHL-S Book Spine Detection | [woody-willis-kly8v/dahl-s-book-spine-detection](https://universe.roboflow.com/woody-willis-kly8v/dahl-s-book-spine-detection) | CC BY 4.0 | `data/raw/woody-willis-dahl-s-book-spine-detection/` |

Rejected during verification (plan §3): `koteitan/book-spine-detection-2cci9`
and `books-26cz6/book-spline-detection` — bounding boxes only, no polygons.

The actual dataset files are gitignored (`data/raw/*/*`) — download raw
(non-augmented) COCO-segmentation exports and place them in the paths
above before running the merge script (plan §2/stage 2).
