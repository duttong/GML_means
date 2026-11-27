# The Hole Story: outreach materials

Classroom materials for high school students that explain CFCs and ozone
depletion with the NOAA/GML annual means produced by this repository.

| File | What it is |
|---|---|
| `index.html` | Interactive web page: chain-reaction animation, a gas explorer with hover values and latitude bands, the CFC-11 "case file", a report card and "read the graph" questions. A single self-contained file. |
| `update_web_data.py` | Refreshes the data embedded in `index.html` from `gml_annual_means/data_files`. |
| `poster/The_Hole_Story_poster_24x36.pdf` | Print-ready poster, 24 × 36 in (vector). The 2:3 shape also prints at 12 × 18 or 16 × 24 in. |
| `poster/The_Hole_Story_poster_24x36.png` | The same poster as a 4608 × 6912 px image. |
| `poster/poster.html` | Poster source (2304 × 3456 px at 96 px/in). |
| `poster/make_pdf.js` | Re-renders the poster PDF and PNG from `poster.html`. |

## Data

Both pieces use the January–December ("jul") yearly files written by
`gml_annualmeans.py`: global means and the four semi-hemispheric means
(HN, LN, LS, HS), 1978–2025 for CFC-11. The web page also includes CFC-12,
CFC-113, CCl4, CH3CCl3, halon-1211, HCFC-22, HCFC-141b and HFC-134a.

Facts not taken from these data files are cited on the page and poster:
gas lifetimes (SPARC 2013; WMO/UNEP 2022) and ozone recovery dates
(WMO/UNEP Scientific Assessment of Ozone Depletion, 2022).

## Updating the web page

After a full run of `gml_annualmeans.py`:

```bash
python docs/update_web_data.py
```

The page's text (for example "212 ppt in 2025" and the timeline) is written by
hand, so check it after adding a new year. The charts, tiles, table and report
card are computed from the embedded data.

The poster's charts were generated once from the 1978–2025 data and are
embedded in `poster.html`; they do not update automatically.

## Publishing with GitHub Pages

In the repository's **Settings → Pages**, choose **Deploy from a branch**, then
select the branch and the `/docs` folder. The page will be served at
`https://<owner>.github.io/GML_means/`.
