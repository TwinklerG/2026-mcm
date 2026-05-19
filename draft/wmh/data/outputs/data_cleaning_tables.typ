// Table 1: Missing value sources (paste into main.typ)
#table(
  columns: (2fr, 4fr, 1.2fr, 2fr),
  inset: 8pt,
  align: (left, left, center, left),
  [*Source*], [*Description*], [*Count*], [*Action*],
  ['Judge slot not used'], ['Fewer judges in some seasons/weeks (e.g. Judge 4 absent)'], [2499], ['Retained (no imputation)'],
  ['Post-elimination'], ['No score recorded after contestant eliminated'], [6904], ['Retained (no imputation)'],
  [], [Total score cells], [18524], [],
  caption: [Missing value sources and handling.],
)

// Table 2: Special score values (paste into main.typ)
#table(
  columns: (1.2fr, 3fr, 1.2fr, 1.5fr),
  inset: 8pt,
  align: (center, left, center, left),
  [*Value*], [*Meaning*], [*Count*], [*Action*],
  ['0'], ['Post-elimination'], [4671], ['Retained'],
  ['> 10'], ['Bonus (competition rule)'], [221], ['Retained'],
  ['(4, 10]'], ['Normal judge score'], [8891], ['Retained'],
  caption: [Special score values and handling.],
)