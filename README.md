# SRE10-SV

## Requirements
* bob.measure
* bob.bio.gmm
* bob.io.base
* bob.learn.em
* bob.bio.spear
* numpy
* sklearn
* dill
* scipy

## Before run
Don't forget to change the data path and trn path in function `get_args()` of sv.py! (Line 170-171)

## How to run
Simple run: `$ python3 sv.py`
with number of test cases = 2, train sep num = 4, output folder path = ''

Args:
`--number_of_test_cases`: int from 1 to 7
`--train_sep_num`: int > 1
`--output_folder_path`: string

## Output
Check result.txt and det_curve.png in your output folder.