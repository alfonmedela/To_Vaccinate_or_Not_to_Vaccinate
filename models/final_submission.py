import pandas as pd

# Load all the submission files
file_1 = pd.read_csv('split_0/submission_file.csv')
file_2 = pd.read_csv('split_1/submission_file.csv')
file_3 = pd.read_csv('split_2/submission_file.csv')
file_4 = pd.read_csv('split_3/submission_file.csv')
file_5 = pd.read_csv('split_4/submission_file.csv')


for i in range(len(file_1)):
    label_1 = file_1['label'][i]
    label_2 = file_2['label'][i]
    label_3 = file_3['label'][i]
    label_4 = file_4['label'][i]
    label_5 = file_5['label'][i]

    joint = (1./5.) * (label_1 + label_2 + label_3 + label_4 + label_5)

    # limit the values to the range [-1, 1]
    if joint > 1.0:
        joint = 1.0
    if joint < -1.0:
        joint = -1.0

    file_1['label'][i] = joint

# Save stacked submission
file_1.to_csv('submission_0_1_2_3_4.csv', index=False)